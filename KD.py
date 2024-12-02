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

# 교사 모델 (더 큰 모델)
class TeacherNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# 학생 모델 (기존의 CircleNet을 약간 수정)
class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
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

# Knowledge Distillation Loss
class DistillationLoss(nn.Module):
    def __init__(self, T=4.0, alpha=0.5):
        super().__init__()
        self.T = T
        self.alpha = alpha
        self.criterion = nn.MSELoss()

    def forward(self, student_outputs, teacher_outputs, targets):
        # Hard Loss - student predictions vs true targets
        hard_loss = self.criterion(student_outputs, targets)
        
        # Soft Loss - student vs teacher predictions
        soft_targets = F.softmax(teacher_outputs / self.T, dim=1)
        soft_predictions = F.softmax(student_outputs / self.T, dim=1)
        soft_loss = F.kl_div(soft_predictions.log(), soft_targets, reduction='batchmean') * (self.T ** 2)
        
        # Combined loss
        loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss
        return loss

def train_with_distillation(teacher_model, student_model, train_loader, device, num_epochs=10):
    teacher_model.eval()  # 교사 모델은 학습된 상태로 고정
    student_model.train()
    
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
    distill_loss = DistillationLoss(T=4.0, alpha=0.5)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            # 교사 모델의 예측
            with torch.no_grad():
                teacher_outputs = teacher_model(imgs)
            
            # 학생 모델의 예측
            student_outputs = student_model(imgs)
            
            # Loss 계산
            loss = distill_loss(student_outputs, teacher_outputs, targets)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
    
    return student_model

def main():
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 데이터 로더 설정
    train_dataset = CircleDataset('train/img', 'train/target')
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # 모델 초기화
    teacher_model = TeacherNet().to(device)
    student_model = StudentNet().to(device)
    
    # 교사 모델 학습 (또는 사전 학습된 모델 로드)
    print("Training teacher model...")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(teacher_model.parameters(), lr=0.001)
    
    for epoch in range(5):  # 교사 모델 학습
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
    
    # Knowledge Distillation 수행
    print("\nStarting Knowledge Distillation...")
    student_model = train_with_distillation(teacher_model, student_model, train_loader, device)
    
    # 학생 모델 저장
    torch.save(student_model.state_dict(), 'KD_model.pth')
    print("Distilled student model saved.")
    
    # 나머지 코드는 기존과 동일하게 유지
    student_model.eval()
    print("Training completed. Starting camera...")
    
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        metrics = measure_performance(frame, student_model, device)
        
        for (x, y, r) in metrics['circles']:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
        
        detected_count = len(metrics['circles'])
        cv2.putText(frame, f"Circles: {detected_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Time: {metrics['inference_time_ms']:.1f}ms", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Memory: {metrics['memory_mb']:.1f}MB", (10, 90),
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