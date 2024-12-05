import cv2
import numpy as np
import time
import psutil
import os
from gtts import gTTS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math

class CircleDataset(Dataset):
    def __init__(self, img_dir, target_dir):
        self.img_dir = img_dir
        self.target_dir = target_dir
        self.img_files = sorted(os.listdir(img_dir))
        self.target_files = sorted(os.listdir(target_dir))
        self.max_circles = 5  

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        target_path = os.path.join(self.target_dir, self.target_files[idx])
        
        img = torch.load(img_path, weights_only=True).float() / 255.0
        targets = torch.load(target_path, weights_only=True)  # [x, y, r] 정보 사용
        
        # Normalize coordinates
        targets[:, 0] = targets[:, 0] / 416  # x coordinate
        targets[:, 1] = targets[:, 1] / 416  # y coordinate
        targets[:, 2] = targets[:, 2] / 416  # radius

        # Padding
        padded_target = torch.zeros(self.max_circles, 3)
        num_circles = min(targets.size(0), self.max_circles)
        padded_target[:num_circles] = targets[:num_circles]
        
        return img.unsqueeze(0), padded_target

class TeacherNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        self.detector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 5 * 3)  # 5개 원의 x, y, r
        )

    def forward(self, x):
        x = self.features(x)
        x = self.detector(x)
        return x.view(-1, 5, 3)

class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  # 채널 수 감소
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        self.detector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 5 * 3)  # Teacher와 동일한 출력
        )

    def forward(self, x):
        x = self.features(x)
        x = self.detector(x)
        return x.view(-1, 5, 3)

class DistillationLoss(nn.Module):
    def __init__(self, T=2.0, alpha=0.3):
        super().__init__()
        self.T = T
        self.alpha = alpha
        self.criterion = nn.MSELoss()

    def forward(self, student_outputs, teacher_outputs, targets):
        # Hard Loss - 실제 타겟과의 MSE
        hard_loss = self.criterion(student_outputs, targets)
        
        # Soft Loss - 교사 모델의 출력과의 MSE
        soft_loss = self.criterion(
            student_outputs / self.T,
            teacher_outputs / self.T
        )
        
        return (1 - self.alpha) * hard_loss + self.alpha * soft_loss

def detect_circles(frame, model, device):
    # 이미지 전처리
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (416, 416))
    img_tensor = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0).to(device) / 255.0
    
    # 모델로 원 검출
    with torch.no_grad():
        predictions = model(img_tensor)[0]  # [5, 3]
    
    # 원의 좌표를 원본 이미지 크기에 맞게 변환
    h, w = frame.shape[:2]
    circles = []
    for x, y, r in predictions.cpu().numpy():
        if r > 0.1:  # confidence threshold
            x = int(x * w)
            y = int(y * h)
            r = int(r * min(w, h))
            circles.append((x, y, r))
    
    return circles, len(circles)

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def measure_performance(frame, model, device, circles=None):
    start_time = time.time()
    circles, num_circles = detect_circles(frame, model, device)
    inference_time = (time.time() - start_time) * 1000

    return {
        'inference_time_ms': inference_time,
        'memory_mb': get_memory_usage(),
        'num_circles': num_circles,
        'circles': circles
    }

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=640,
    capture_height=480,
    display_width=640,
    display_height=480,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
    

def train_with_distillation(teacher_model, student_model, train_loader, device, num_epochs=2):
    teacher_model.eval()
    student_model.train()
    
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
    distill_criterion = DistillationLoss()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for imgs, targets in train_loader:  # targets: [batch_size, 5, 3]
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            with torch.no_grad():
                teacher_outputs = teacher_model(imgs)  # [batch_size, 5, 3]
            
            student_outputs = student_model(imgs)  # [batch_size, 5, 3]
            loss = distill_criterion(student_outputs, teacher_outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
    
    return student_model

def main():
    device = torch.device("cpu") 
    print(f"Using device: {device}")
    
    # 데이터 로더
    train_dataset = CircleDataset('train/img', 'train/target')
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # 모델 초기화
    teacher_model = TeacherNet().to(device)
    student_model = StudentNet().to(device)
    
    print("Training teacher model...")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(teacher_model.parameters(), lr=0.001)
    
    for epoch in range(2):
        teacher_model.train()
        running_loss = 0.0
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = teacher_model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Teacher Model - Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
    
    # Knowledge Distillation
    print("\nStarting Knowledge Distillation...")
    distill_criterion = DistillationLoss()
    student_optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
    
    for epoch in range(2):
        student_model.train()
        running_loss = 0.0
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            with torch.no_grad():
                teacher_outputs = teacher_model(imgs)
            
            student_outputs = student_model(imgs)
            loss = distill_criterion(student_outputs, teacher_outputs, targets)
            
            student_optimizer.zero_grad()
            loss.backward()
            student_optimizer.step()
            
            running_loss += loss.item()
        print(f"Student Model - Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
    
    # 모델 저장
    torch.save(student_model.state_dict(), 'KD_model.pth')
    print("Distilled student model saved.")
    
    # 카메라 실행
    student_model.eval()
    print("Training completed. Starting camera...")
    
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        metrics = measure_performance(frame, student_model, device, None)
        
        for (x, y, r) in metrics['circles']:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
        
        detected_count = len(metrics['circles'])
        cv2.putText(frame, f"Circles: {detected_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 정보 표시
        cv2.putText(frame, f"Circles: {metrics['num_circles']}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Time: {metrics['inference_time_ms']:.1f}ms", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Memory: {metrics['memory_mb']:.1f}MB", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'c' to count circles", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("Circles", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            text = f"{metrics['num_circles']}개의 원이 있습니다."
            tts = gTTS(text=text, lang='ko')
            tts.save("circles.wav")
            os.system("start circles.wav")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
