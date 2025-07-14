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
        distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        if distance < (r1 + r2):
            return True
    return False

def generate_random_circle_image(size=416, max_attempts=100):
    num_circles = torch.randint(1, 6, (1,)).item() 
    img = torch.zeros(size, size)
    targets = []

    for _ in range(num_circles):
        attempt = 0
        while attempt < max_attempts:
            x = torch.randint(50, size - 50, (1,)).item()
            y = torch.randint(50, size - 50, (1,)).item()
            r = torch.randint(10, 30, (1,)).item()
            
            new_circle = [x, y, r]
            
            if not targets or not check_circle_overlap(new_circle, targets):
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

if not os.path.exists('train'):
    os.makedirs('train')
if not os.path.exists('train/img'):
    os.makedirs('train/img')
if not os.path.exists('train/target'):
    os.makedirs('train/target')

for i in range(100):
    img, targets = generate_random_circle_image()
    torch.save(torch.tensor(img), f'train/img/circle_{i+1}.pt')
    torch.save(targets, f'train/target/circle_{i+1}.pt')
