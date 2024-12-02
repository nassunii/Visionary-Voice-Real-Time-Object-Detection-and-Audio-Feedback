import cv2
import numpy as np
import time
import psutil
import os
from gtts import gTTS

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB 단위

def measure_performance(frame, circles):
    # 실행 시간 측정
    start_time = time.time()
    circles = detect_circles(frame)
    inference_time = (time.time() - start_time) * 1000  # ms 단위

    # 메모리 사용량
    memory_usage = get_memory_usage()

    # FLOPs 계산 (근사치)
    h, w = frame.shape[:2]
    approx_flops = h * w * 9  # 기본 연산 추정치

    return {
        'inference_time_ms': inference_time,
        'memory_mb': memory_usage,
        'flops': approx_flops
    }

def merge_circles(circles, distance_threshold=30):
    merged = []
    used = set()
    
    for i, (x1, y1, r1) in enumerate(circles):
        if i in used:
            continue
            
        current_circle = [x1, y1, r1]
        count = 1
        
        for j, (x2, y2, r2) in enumerate(circles[i+1:], i+1):
            if j in used:
                continue
                
            distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if distance < distance_threshold:
                current_circle[0] += x2
                current_circle[1] += y2
                current_circle[2] += r2
                count += 1
                used.add(j)
                
        if count > 1:
            current_circle = [int(c/count) for c in current_circle]
            
        if i not in used:
            merged.append(tuple(current_circle))
            
    return merged

def detect_circles(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
    
    return merge_circles(circles)

cap = cv2.VideoCapture(0)

metrics_list = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    circles = detect_circles(frame)
    metrics = measure_performance(frame, circles)
    metrics_list.append(metrics)

    # 화면에 성능 지표 표시
    cv2.putText(frame, f"Time: {metrics['inference_time_ms']:.1f}ms", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Memory: {metrics['memory_mb']:.1f}MB", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, "Press 'c' to count circles", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    for (x, y, r) in circles:
        cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
        cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
    
    cv2.putText(frame, f"Circles: {len(circles)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Circles", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        text = f"{len(circles)}개의 원이 있습니다."
        tts = gTTS(text=text, lang='ko')
        tts.save("circles.mp3")
        os.system("start circles.mp3")

cap.release()
cv2.destroyAllWindows()