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

class CircleNet_FP(nn.Module):
    def __init__(self, pruning_ratio=0.5):
        super().__init__()
        # 초기 채널 수 정의
        self.channels = {
            'conv1': 16,
            'conv2': 32,
            'conv3': 64
        }
        self.pruning_ratio = pruning_ratio
        
        # 초기 모델 구성
        self.features = nn.Sequential(
            nn.Conv2d(1, self.channels['conv1'], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(self.channels['conv1'], self.channels['conv2'], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(self.channels['conv2'], self.channels['conv3'], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.channels['conv3'] * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
        
    def get_filter_importance(self, conv_layer):
        # L1-norm을 사용하여 각 필터의 중요도 계산
        importance = torch.sum(torch.abs(conv_layer.weight.data), dim=(1, 2, 3))
        return importance

    def prune_filters(self):
        new_model = CircleNet_FP(self.pruning_ratio)
        
        # 각 conv layer에 대해 필터 중요도 계산 및 pruning
        conv_layers = [module for module in self.modules() if isinstance(module, nn.Conv2d)]
        new_conv_layers = [module for module in new_model.modules() if isinstance(module, nn.Conv2d)]
        
        prev_remaining_filters = 1  # 입력 채널은 1
        
        for i, (conv, new_conv) in enumerate(zip(conv_layers, new_conv_layers)):
            importance = self.get_filter_importance(conv)
            num_filters = importance.size(0)
            num_keep = int(num_filters * (1 - self.pruning_ratio))
            
            # 중요도가 높은 필터의 인덱스 선택
            _, indices = torch.sort(importance, descending=True)
            keep_indices = indices[:num_keep]
            
            # 선택된 필터만 새 모델로 복사
            new_conv.weight.data = conv.weight.data[keep_indices][:, :prev_remaining_filters]
            if conv.bias is not None:
                new_conv.bias.data = conv.bias.data[keep_indices]
            
            prev_remaining_filters = num_keep
            
            # channels 업데이트
            if i == 0:
                self.channels['conv1'] = num_keep
            elif i == 1:
                self.channels['conv2'] = num_keep
            else:
                self.channels['conv3'] = num_keep
        
        # Classifier의 입력 차원 조정
        in_features = self.channels['conv3'] * 7 * 7
        new_model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
        
        return new_model

def get_memory_usage():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

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
    
    circles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > 0.7 and area > 100:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                circles.append((int(x), int(y), int(radius)))
    
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
                
            distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if distance < max(r1, r2):
                current[0] += x2
                current[1] += y2
                current[2] += r2
                count += 1
                used.add(j)
                
        if count > 1:
            current = [int(c/count) for c in current]
            
        if i not in used:
            merged_circles.append(tuple(current))
    
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

def main():
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # 학습 데이터 로드
    train_dataset = CircleDataset('train/img', 'train/target')
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # 초기 모델 생성
    model = CircleNet_FP(pruning_ratio=0.5).to(device)
    
    # 학습 전 필터 pruning 수행
    model = model.prune_filters()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss().to(device)

    # 학습
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
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), 'FP_model.pth')
    print("Trained model saved.")

    model.eval()
    print("Training completed. Starting camera...")
    
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        metrics = measure_performance(frame, model, device, None)
        
        for (x, y, r) in metrics['circles']:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
        
        detected_count = len(metrics['circles'])
        cv2.putText(frame, f"Circles: {detected_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
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