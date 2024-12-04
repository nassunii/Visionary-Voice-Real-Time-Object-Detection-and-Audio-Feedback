import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import CircleDataset
from pruning import CircleNet_P
from FP import CircleNet_FP

def train_and_save_models(pruning_ratios=[0.25, 0.50, 0.75]):
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # 학습 데이터 로드 (detection dataset)
    train_dataset = CircleDataset('train/img', 'train/target')
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # 각 pruning ratio에 대해 두 가지 방식의 모델 학습
    for ratio in pruning_ratios:
        print(f"\nTraining models with pruning ratio: {ratio}")
        
        # 1. Filter Pruning Model
        print("Training Filter Pruning Model...")
        model_fp = CircleNet_FP(pruning_ratio=ratio).to(device)
        model_fp = model_fp.prune_filters()  # 필터 프루닝 적용
        
        optimizer = torch.optim.Adam(model_fp.parameters(), lr=0.001)
        criterion = nn.MSELoss().to(device)

        # 학습
        for epoch in range(2):
            model_fp.train()
            running_loss = 0.0
            for imgs, targets in train_loader:  # targets: [batch_size, 5, 3]
                imgs = imgs.to(device)
                targets = targets.to(device)
                
                optimizer.zero_grad()
                outputs = model_fp(imgs)  # [batch_size, 5, 3]
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"FP Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")

        # 모델 저장
        torch.save(model_fp.state_dict(), f'FP_model_{int(ratio*100)}.pth')
        print(f"Filter Pruning model saved with {int(ratio*100)}% ratio")

        # 2. Weight Pruning Model
        print("\nTraining Weight Pruning Model...")
        model_p = CircleNet_P(pruning_ratio=ratio).to(device)
        model_p.apply_pruning()  # 가중치 프루닝 적용
        
        optimizer = torch.optim.Adam(model_p.parameters(), lr=0.001)

        # 학습
        for epoch in range(2):
            model_p.train()
            running_loss = 0.0
            for imgs, targets in train_loader:  # targets: [batch_size, 5, 3]
                imgs = imgs.to(device)
                targets = targets.to(device)
                
                optimizer.zero_grad()
                outputs = model_p(imgs)  # [batch_size, 5, 3]
                loss = criterion(outputs, targets)
                loss.backward()

                # 그라디언트 업데이트 전 마스크 적용
                for name, module in model_p.named_modules():
                    if isinstance(module, (nn.Conv2d, nn.Linear)):
                        if hasattr(module, 'weight_grad'):
                            module.weight.grad *= model_p.masks[name]

                optimizer.step()

                # Weight pruning 마스크 재적용
                for name, module in model_p.named_modules():
                    if isinstance(module, (nn.Conv2d, nn.Linear)):
                        module.weight.data *= model_p.masks[name]

                running_loss += loss.item()
            print(f"P Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")

        # 모델 저장
        torch.save(model_p.state_dict(), f'P_model_{int(ratio*100)}.pth')
        print(f"Weight Pruning model saved with {int(ratio*100)}% ratio")

if __name__ == "__main__":
    train_and_save_models([0.25, 0.50, 0.75])