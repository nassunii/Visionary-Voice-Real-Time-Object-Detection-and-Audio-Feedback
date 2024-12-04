import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import time
import psutil
from gtts import gTTS

class CircleDataset(Dataset):
    def __init__(self, img_dir, target_dir, max_circles=5):
        self.img_dir = img_dir
        self.target_dir = target_dir
        self.img_files = sorted(os.listdir(img_dir))
        self.target_files = sorted(os.listdir(target_dir))
        self.max_circles = max_circles

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        target_path = os.path.join(self.target_dir, self.target_files[idx])
        
        img = torch.load(img_path).float() / 255.0
        targets = torch.load(target_path)
        
        # Normalize coordinates to [0, 1] range
        targets[:, 0] = targets[:, 0] / 416  # x coordinate
        targets[:, 1] = targets[:, 1] / 416  # y coordinate
        targets[:, 2] = targets[:, 2] / 416  # radius
        
        # Pad targets to max_circles
        padded_targets = torch.zeros(self.max_circles, 3)
        num_circles = min(targets.size(0), self.max_circles)
        padded_targets[:num_circles] = targets[:num_circles]
        
        return img.unsqueeze(0), padded_targets

class CircleNet(nn.Module):
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
            nn.Linear(128, 5 * 3)  # 5 circles * (x, y, r)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.detector(x)
        return x.view(-1, 5, 3)  # reshape to [batch_size, 5, 3]

def preprocess_frame(frame):
    """카메라 프레임을 모델 입력용으로 전처리"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Adaptive thresholding to handle different lighting conditions
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    
    # Resize to model input size
    resized = cv2.resize(thresh, (416, 416))
    return resized

def detect_circles(frame, model, device, conf_threshold=0.1):
    # 이미지 전처리
    processed = preprocess_frame(frame)
    img_tensor = torch.FloatTensor(processed).unsqueeze(0).unsqueeze(0).to(device) / 255.0
    
    # 모델 추론
    with torch.no_grad():
        predictions = model(img_tensor)[0]  # [5, 3]
    
    # Convert normalized coordinates back to image coordinates
    circles = []
    h, w = frame.shape[:2]
    
    for x, y, r in predictions.cpu().numpy():
        # Confidence check using radius
        if r > conf_threshold:
            # Convert normalized coordinates back to pixel coordinates
            x_pixel = int(x * w)
            y_pixel = int(y * h)
            r_pixel = int(r * min(w, h))
            
            circles.append((x_pixel, y_pixel, r_pixel))
    
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

def main():
    # 모델 및 학습 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 데이터 로드 및 모델 초기화
    train_dataset = CircleDataset('train/img', 'train/target')
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    model = CircleNet().to(device)
    
    # 학습
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print("Starting training...")
    for epoch in range(2):  
        model.train()
        running_loss = 0.0
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

    print("Training completed. Saving model...")
    torch.save(model.state_dict(), 'Basic_model.pth')
    
    model.eval()
    print("Starting camera...")
    
    # 카메라 설정
    try:
        cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    except:
        print("Falling back to regular camera...")
        cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 성능 측정 및 원 검출
        metrics = measure_performance(frame, model, device)
        
        # 검출된 원 그리기
        for (x, y, r) in metrics['circles']:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)  # 원 둘레
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)  # 중심점
        
        # 정보 표시
        cv2.putText(frame, f"Circles: {metrics['num_circles']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Time: {metrics['inference_time_ms']:.1f}ms", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Memory: {metrics['memory_mb']:.1f}MB", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("Circle Detection", frame)
        
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