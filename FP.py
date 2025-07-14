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
        self.max_circles = 5

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        target_path = os.path.join(self.target_dir, self.target_files[idx])
        
        # weights_only=True 추가
        img = torch.load(img_path, weights_only=True).float() / 255.0
        targets = torch.load(target_path, weights_only=True)  # [x, y, r] 정보
        
        # Normalize coordinates
        targets[:, 0] = targets[:, 0] / 416  # x coordinate
        targets[:, 1] = targets[:, 1] / 416  # y coordinate
        targets[:, 2] = targets[:, 2] / 416  # radius

        # Padding for fixed size output
        padded_target = torch.zeros(self.max_circles, 3)  # 5개의 원까지 가능
        num_circles = min(targets.size(0), self.max_circles)
        padded_target[:num_circles] = targets[:num_circles]
        
        return img.unsqueeze(0), padded_target

class CircleNet_FP(nn.Module):
    def __init__(self, pruning_ratio=0.5):
        super().__init__()
        self.channels = {
            'conv1': 16,
            'conv2': 32,
            'conv3': 64
        }
        self.pruning_ratio = pruning_ratio
        
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
        
        self.detector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.channels['conv3'] * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 5 * 3)  # 5개 원의 x, y, r
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.detector(x)
        return x.view(-1, 5, 3)  # [batch_size, 5, 3] 형태로 반환
        
    def get_filter_importance(self, conv_layer):
        importance = torch.sum(torch.abs(conv_layer.weight.data), dim=(1, 2, 3))
        return importance

    def prune_filters(self):
        new_model = CircleNet_FP(self.pruning_ratio)
        conv_layers = [module for module in self.modules() if isinstance(module, nn.Conv2d)]
        new_conv_layers = [module for module in new_model.modules() if isinstance(module, nn.Conv2d)]
        
        prev_remaining_filters = 1  # 입력 채널 1
        for i, (conv, new_conv) in enumerate(zip(conv_layers, new_conv_layers)):
            importance = self.get_filter_importance(conv)
            num_filters = importance.size(0)
            num_keep = int(num_filters * (1 - self.pruning_ratio))

            _, indices = torch.sort(importance, descending=True)
            keep_indices = indices[:num_keep]
            new_conv.weight.data = conv.weight.data[keep_indices][:, :prev_remaining_filters]
            if conv.bias is not None:
                new_conv.bias.data = conv.bias.data[keep_indices]
            
            prev_remaining_filters = num_keep
            
            if i == 0:
                new_model.channels['conv1'] = num_keep
            elif i == 1:
                new_model.channels['conv2'] = num_keep
            else:
                new_model.channels['conv3'] = num_keep

        in_features = new_model.channels['conv3'] * 7 * 7
        new_model.detector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 5 * 3)
        )
        
        return new_model

def detect_circles(frame, model, device):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (416, 416))
    img_tensor = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0).to(device) / 255.0
    with torch.no_grad():
        predictions = model(img_tensor)[0]  # [5, 3]
    
    h, w = frame.shape[:2]
    circles = []
    for x, y, r in predictions.cpu().numpy():
        if r > 0.1:  # confidence threshold
            x = int(x * w)
            y = int(y * h)
            r = int(r * min(w, h))
            circles.append((x, y, r))
    
    return circles, len(circles)

def main():
    device = torch.device("cpu")
    print(f"Using device: {device}")
    train_dataset = CircleDataset('train/img', 'train/target')
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    model = CircleNet_FP(pruning_ratio=0.5).to(device)
    model = model.prune_filters()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss().to(device)

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
    device = torch.device("cpu")
    print(f"Using device: {device}")
    train_dataset = CircleDataset('train/img', 'train/target')
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    model = CircleNet_FP(pruning_ratio=0.5).to(device)
    model = model.prune_filters()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss().to(device)

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
