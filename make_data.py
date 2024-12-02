import torch
import os
import matplotlib.pyplot as plt
import numpy as np

def check_circle_overlap(new_circle, existing_circles):
    """
    새로운 원이 기존 원들과 겹치는지 확인
    new_circle: [x, y, r]
    existing_circles: list of [x, y, r]
    """
    x1, y1, r1 = new_circle
    
    for circle in existing_circles:
        x2, y2, r2 = circle
        # 두 원의 중심점 사이의 거리 계산
        distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        # 두 원의 반지름 합보다 거리가 작으면 겹침
        if distance < (r1 + r2):
            return True
    return False

def generate_random_circle_image(size=416, max_attempts=100):
    num_circles = torch.randint(1, 6, (1,)).item()  # 1~5개의 원 생성
    img = torch.zeros(size, size)
    targets = []

    for _ in range(num_circles):
        attempt = 0
        while attempt < max_attempts:
            x = torch.randint(50, size - 50, (1,)).item()
            y = torch.randint(50, size - 50, (1,)).item()
            r = torch.randint(10, 30, (1,)).item()
            
            new_circle = [x, y, r]
            
            # 첫 번째 원이거나 기존 원들과 겹치지 않는 경우
            if not targets or not check_circle_overlap(new_circle, targets):
                # 원 그리기
                for i in range(size):
                    for j in range(size):
                        if (i - y) ** 2 + (j - x) ** 2 <= r ** 2:
                            img[i, j] = 1
                
                targets.append(new_circle)
                break
                
            attempt += 1
        
        if attempt >= max_attempts:
            print(f"Warning: Could not place circle {len(targets)+1} without overlap")
            break

    return img.numpy(), torch.tensor(targets)

def visualize_image(img):
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

# Directory setup
if not os.path.exists('train'):
    os.makedirs('train')
if not os.path.exists('train/img'):
    os.makedirs('train/img')
if not os.path.exists('train/target'):
    os.makedirs('train/target')

# Generate training data
for i in range(100):
    img, targets = generate_random_circle_image()
    #visualize_image(img)  # 생성된 이미지 확인시 주석 해제
    torch.save(torch.tensor(img), f'train/img/circle_{i+1}.pt')
    torch.save(targets, f'train/target/circle_{i+1}.pt')