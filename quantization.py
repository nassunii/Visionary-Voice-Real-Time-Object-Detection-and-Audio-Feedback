import cv2
import numpy as np
import time
import psutil
import os
from gtts import gTTS
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class CircleDataset(Dataset):
    def __init__(self, img_dir, target_dir, enable_fp16=True):
        self.img_dir = img_dir
        self.target_dir = target_dir
        self.img_files = sorted(os.listdir(img_dir))
        self.target_files = sorted(os.listdir(target_dir))
        self.enable_fp16 = enable_fp16
        self.max_circles = 5

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        target_path = os.path.join(self.target_dir, self.target_files[idx])
        
        img = torch.load(img_path, weights_only=True).float() / 255.0
        targets = torch.load(target_path, weights_only=True)  # [x, y, r] 정보

        # Normalize coordinates
        targets[:, 0] = targets[:, 0] / 416  # x coordinate
        targets[:, 1] = targets[:, 1] / 416  # y coordinate
        targets[:, 2] = targets[:, 2] / 416  # radius

        # Padding
        padded_target = torch.zeros(self.max_circles, 3)
        num_circles = min(targets.size(0), self.max_circles)
        padded_target[:num_circles] = targets[:num_circles]
        
        if self.enable_fp16:
            img = img.half()
            padded_target = padded_target.half()
        
        return img.unsqueeze(0), padded_target

train_dataset = CircleDataset('train/img', 'train/target')
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

class CircleNet_Q(nn.Module):
    def __init__(self, enable_fp16=True):
        super().__init__()
        self.enable_fp16 = enable_fp16 and torch.cuda.is_available()
        
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
        
        self.apply(self._init_weights)
        
        if self.enable_fp16:
            self.half()
            
        if torch.cuda.is_available():
            self.cuda()
            torch.backends.cudnn.benchmark = True
            
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            if self.enable_fp16:
                m.weight.data = m.weight.data.half()
                if m.bias is not None:
                    m.bias.data = m.bias.data.half()
    
    def forward(self, x):
        if self.enable_fp16 and x.dtype != torch.float16:
            x = x.half()
        
        with autocast(enabled=self.enable_fp16):
            x = self.features(x)
            x = self.detector(x)
            x = x.view(-1, 5, 3)  # reshape to [batch_size, 5, 3]
        
        return x

def detect_circles(frame, model, device, enable_fp16=True):
    enable_fp16 = enable_fp16 and torch.cuda.is_available()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (416, 416))
    img_tensor = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0).to(device) / 255.0
    
    if enable_fp16:
        img_tensor = img_tensor.half()

    with torch.no_grad():
        with autocast(enabled=enable_fp16):
            predictions = model(img_tensor)[0]  # [5, 3]

    h, w = frame.shape[:2]
    circles = []
    for x, y, r in predictions.cpu().float().numpy():
        if r > 0.1:  # confidence threshold
            x = int(x * w)
            y = int(y * h)
            r = int(r * min(w, h))
            circles.append((x, y, r))
    
    return circles, len(circles)

class CircleTrainer:
    def __init__(self, model, criterion, optimizer, device, enable_fp16=True):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.enable_fp16 = enable_fp16 and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.enable_fp16)
    
    def train_step(self, inputs, targets):  # targets: [batch_size, 5, 3]
        self.model.train()
        self.optimizer.zero_grad()
        
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        if self.enable_fp16:
            inputs = inputs.half()
            targets = targets.half()
        
        with autocast(enabled=self.enable_fp16):
            outputs = self.model(inputs)  # [batch_size, 5, 3]
            loss = self.criterion(outputs, targets)
        
        if self.enable_fp16:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        
        return loss.item()

def get_memory_usage():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

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
    enable_fp16 = False
    device = torch.device("cpu")
    print(f"Using device: {device}")

    train_dataset = CircleDataset('train/img', 'train/target', enable_fp16=enable_fp16)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    model = CircleNet_Q(enable_fp16=True).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = CircleTrainer(model, criterion, optimizer, device, enable_fp16=enable_fp16)

    num_epochs = 2
    for epoch in range(num_epochs):
        running_loss = 0.0
        for imgs, targets in train_loader:
            loss = trainer.train_step(imgs, targets)
            running_loss += loss
            
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), 'Q_model.pth')
    print("Trained model saved.")
    model.eval()
    print("Training completed. Starting camera...")
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        metrics = measure_performance(frame, model, device)
        for (x, y, r) in metrics['circles']:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

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
            os.system("aplay circles.wav") 

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
