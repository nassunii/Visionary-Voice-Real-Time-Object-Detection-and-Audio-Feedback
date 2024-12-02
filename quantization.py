import cv2
import torch
import torch.nn as nn
import torch.quantization
import time
import psutil
import os
from gtts import gTTS

class CircleNet_Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 104 * 104, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        for module in self.modules():
            if type(module) == nn.Sequential:
                for i, m in enumerate(module):
                    if type(m) == nn.Conv2d and \
                       i + 1 < len(module) and \
                       type(module[i + 1]) == nn.ReLU:
                        torch.quantization.fuse_modules(module, [str(i), str(i + 1)], inplace=True)

def prepare_quantization(model):
    model.eval()
    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    return model

def convert_to_quantized(model):
    torch.quantization.convert(model, inplace=True)
    return model

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
    # CPU 사용 (quantization은 CPU에서만 지원)
    device = torch.device("cpu")
    
    # 모델 생성 및 양자화
    model = CircleNet_Q().to(device)
    model = prepare_quantization(model)
    model = convert_to_quantized(model)
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        metrics = measure_performance(frame, model)
        
        # 화면에 정보 표시
        info_text = [
            f"Model: Quantized",
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