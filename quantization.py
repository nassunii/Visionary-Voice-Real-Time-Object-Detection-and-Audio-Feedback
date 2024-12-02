import cv2
import numpy as np
import time
import psutil
import os
from gtts import gTTS
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


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


class CircleNet_Q_Manual(nn.Module):
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
        # Quantization
        x = torch.quantize_per_tensor(x, scale=0.1, zero_point=128, dtype=torch.quint8)
        
        # Dequantization for processing in layers
        x = x.dequantize()
        x = self.features(x)
        x = self.classifier(x)
        
        return x


def prepare_quantization_manual(model, calibration_loader):
    model.eval()
    # Calibration step
    with torch.no_grad():
        for images, _ in calibration_loader:
            # Forward pass to simulate real quantization
            model(images)
    return model


def get_memory_usage():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def detect_circles(frame, model, device):
    # 이미지 전처리
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (416, 416))
    img_tensor = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        pred = model(img_tensor)

    return [], 0  # Placeholder for actual detection


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


def main():
    # CPU 사용
    device = torch.device("cpu")
    print(f"Using device: {device}")

    calibration_size = len(train_dataset) // 10
    calibration_indices = torch.randperm(len(train_dataset))[:calibration_size]
    calibration_dataset = torch.utils.data.Subset(train_dataset, calibration_indices)
    calibration_loader = DataLoader(calibration_dataset, batch_size=32, shuffle=False)
    
    model = CircleNet_Q_Manual().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss().to(device)

    # 학습
    for epoch in range(2):
        model.train()
        running_loss = 0.0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")
    
    model = prepare_quantization_manual(model, calibration_loader)

    torch.save(model.state_dict(), 'Q_model.pth')
    print("Trained model saved.")

    model.eval()
    print("Training completed. Starting camera...")
    
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        metrics = measure_performance(frame, model, device, None)
        
        # 검출된 원 그리기
        for (x, y, r) in metrics['circles']:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
        
        detected_count = len(metrics['circles'])
        cv2.putText(frame, f"Circles: {detected_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
