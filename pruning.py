import cv2
import torch
import torch.nn as nn
import time
import psutil
import os
from gtts import gTTS

class CircleNet_P(nn.Module):
    def __init__(self, pruning_ratio=0.5):
        super().__init__()
        self.pruning_ratio = pruning_ratio
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 104 * 104, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.masks = {}
        
    def apply_pruning(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                mask = torch.ones_like(module.weight.data)
                tensor_flat = torch.abs(module.weight.data).flatten()
                k = int(len(tensor_flat) * self.pruning_ratio)
                if k > 0:
                    threshold = tensor_flat.kthvalue(k)[0]
                    mask[torch.abs(module.weight.data) <= threshold] = 0
                self.masks[name] = mask
                module.weight.data *= mask
                
    def forward(self, x):
        return self.model(x)

def get_memory_usage():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

def detect_circles(frame, model):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (416, 416))
    img_tensor = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0) / 255.0
    
    with torch.no_grad():
        pred = model(img_tensor)
    return int(pred.item())

def measure_performance(frame, model):
    start_time = time.time()
    circles = detect_circles(frame, model)
    inference_time = (time.time() - start_time) * 1000

    return {
        'inference_time_ms': inference_time,
        'memory_mb': get_memory_usage(),
        'num_circles': circles
    }

def main():
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Pruned 모델 생성 및 초기화
    model = CircleNet_P(pruning_ratio=0.5).to(device)
    model.apply_pruning()
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        metrics = measure_performance(frame, model)
        
        info_text = [
            f"Model: Pruned 50%",
            f"Circles: {metrics['num_circles']}",
            f"Time: {metrics['inference_time_ms']:.1f}ms",
            f"Memory: {metrics['memory_mb']:.1f}MB",
            "Press 'c' to count circles, 'q' to quit"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (10, 30 + i*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("Circle Detection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            text = f"{metrics['num_circles']}개의 원이 있습니다."
            tts = gTTS(text=text, lang='ko')
            tts.save("circles.mp3")
            os.system("start circles.mp3")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()