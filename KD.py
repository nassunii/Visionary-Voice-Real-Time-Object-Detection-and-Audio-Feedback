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

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        target_path = os.path.join(self.target_dir, self.target_files[idx])
        
        img = torch.load(img_path).float() / 255.0
        target = torch.load(target_path).size(0)
        
        return img.unsqueeze(0), torch.tensor([target], dtype=torch.float32)


# 학습 데이터 로드
train_dataset = CircleDataset('train/img', 'train/target')
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)


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
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def detect_circles(frame, model, device):
    # 이미지 전처리
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (416, 416))
    img_tensor = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0).to(device) / 255.0
    
    # OpenCV로 원의 위치 검출
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # 원 후보 검출
    circles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > 0.7 and area > 100:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                circles.append((int(x), int(y), int(radius)))
    
    # 근접한 원들 병합
    merged_circles = []
    used = set()
    
    for i, (x1, y1, r1) in enumerate(circles):
        if i in used:
            continue
            
        current = [x1, y1, r1]
        count = 1
        
        for j, (x2, y2, r2) in enumerate(circles[i+1:], i+1):
            if j in used:
                continue
                
            # 두 원의 중심점 사이 거리 계산
            distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if distance < max(r1, r2):  # 원이 겹치는 경우
                current[0] += x2
                current[1] += y2
                current[2] += r2
                count += 1
                used.add(j)
                
        if count > 1:
            current = [int(c/count) for c in current]
            
        if i not in used:
            merged_circles.append(tuple(current))
    
    # 딥러닝 모델로 검증 (필요한 경우)
    with torch.no_grad():
        pred = model(img_tensor)
    
    return merged_circles, len(merged_circles) 


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

class DistillationLoss(nn.Module):
    def __init__(self, T=2.0, alpha=0.3):
        super().__init__()
        self.T = T
        self.alpha = alpha
        self.criterion = nn.MSELoss()

    def forward(self, student_outputs, teacher_outputs, targets):
        # Hard Loss
        hard_loss = self.criterion(student_outputs, targets)
        
        # Soft Loss
        soft_loss = nn.MSELoss()(
            F.softmax(student_outputs / self.T, dim=1),
            F.softmax(teacher_outputs / self.T, dim=1)
        )
        
        return (1 - self.alpha) * hard_loss + self.alpha * soft_loss
    

def train_with_distillation(teacher_model, student_model, train_loader, device, num_epochs=2):
    teacher_model.eval()
    student_model.train()
    
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
    distill_loss = DistillationLoss()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            with torch.no_grad():
                teacher_outputs = teacher_model(imgs)
            
            student_outputs = student_model(imgs)
            loss = distill_loss(student_outputs, teacher_outputs, targets)
            
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
    
    # 교사 모델 학습
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
    student_model = train_with_distillation(teacher_model, student_model, train_loader, device)
    
    # 모델 저장
    torch.save(student_model.state_dict(), 'KD_lightweight_model.pth')
    print("Distilled student model saved.")
    
    # 카메라 실행 및 추론
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
