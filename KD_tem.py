import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import CircleDataset
from KD import TeacherNet, StudentNet, DistillationLoss

def train_and_save_kd_models(temperatures=[1.0, 2.0, 3.0, 4.0]):
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load training data
    train_dataset = CircleDataset('train/img', 'train/target')
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Initialize and train teacher model once
    print("\nTraining teacher model...")
    teacher_model = TeacherNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(teacher_model.parameters(), lr=0.001)

    for epoch in range(2):
        teacher_model.train()
        running_loss = 0.0
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = teacher_model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Teacher Model - Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

    # Save teacher model
    torch.save(teacher_model.state_dict(), 'teacher_model.pth')
    
    # Train student models with different temperatures
    teacher_model.eval()
    for temp in temperatures:
        print(f"\nTraining student model with temperature: {temp}")
        
        # Initialize student model
        student_model = StudentNet().to(device)
        distill_criterion = DistillationLoss(T=temp)
        student_optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
        
        # Training loop
        for epoch in range(2):
            student_model.train()
            running_loss = 0.0
            for imgs, targets in train_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                
                with torch.no_grad():
                    teacher_outputs = teacher_model(imgs)
                
                student_outputs = student_model(imgs)
                loss = distill_criterion(student_outputs, teacher_outputs, targets)
                
                student_optimizer.zero_grad()
                loss.backward()
                student_optimizer.step()
                
                running_loss += loss.item()
            print(f"Student Model (T={temp}) - Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
        
        # Save student model
        torch.save(student_model.state_dict(), f'KD_model_T{temp}.pth')
        print(f"Student model saved with temperature {temp}")

if __name__ == "__main__":
    train_and_save_kd_models([1.0, 2.0, 3.0, 4.0])