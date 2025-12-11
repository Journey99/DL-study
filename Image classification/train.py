import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm
import json

# ============================================
# 1. 커스텀 데이터셋 정의
# ============================================
class CustomImageDataset(Dataset):
    """
    데이터셋 구조:
    root/
        class1/
            img1.jpg
            img2.jpg
        class2/
            img1.jpg
            img2.jpg
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # 클래스 이름과 인덱스 매핑
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # 모든 이미지 경로와 레이블 수집
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            class_idx = self.class_to_idx[class_name]
            
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append((img_path, class_idx))
        
        print(f"데이터셋 로드 완료:")
        print(f"  - 클래스 수: {len(self.classes)}")
        print(f"  - 총 이미지 수: {len(self.samples)}")
        print(f"  - 클래스: {self.classes}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ============================================
# 2. 데이터 전처리
# ============================================
def get_transforms(img_size=224, is_training=True):
    """
    학습/검증용 transform
    """
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),  # 256 for 224
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

# ============================================
# 3. 모델 생성 (Transfer Learning)
# ============================================
def create_model(num_classes, model_name='efficientnet_b3', pretrained=True):
    """
    EfficientNet 모델 생성 및 분류기 교체
    
    Args:
        num_classes: 커스텀 데이터셋의 클래스 수
        model_name: timm 모델 이름
        pretrained: ImageNet pretrained weight 사용 여부
    """
    # Pretrained 모델 로드
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    
    print(f"\n모델 생성 완료: {model_name}")
    print(f"  - Pretrained: {pretrained}")
    print(f"  - Output classes: {num_classes}")
    
    return model

# ============================================
# 4. 학습 함수
# ============================================
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # 통계
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Progress bar 업데이트
        pbar.set_postfix({
            'loss': f'{running_loss/len(dataloader):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

# ============================================
# 5. 검증 함수
# ============================================
def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/len(dataloader):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

# ============================================
# 6. 메인 학습 파이프라인
# ============================================
def train_model(
    train_dir,
    val_dir,
    model_name='efficientnet_b3',
    num_epochs=50,
    batch_size=32,
    learning_rate=1e-4,
    img_size=224,
    save_dir='checkpoints'
):
    """
    전체 학습 파이프라인
    """
    # Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 데이터셋 로드
    train_dataset = CustomImageDataset(
        train_dir, 
        transform=get_transforms(img_size, is_training=True)
    )
    val_dataset = CustomImageDataset(
        val_dir, 
        transform=get_transforms(img_size, is_training=False)
    )
    
    # 클래스 정보 저장
    os.makedirs(save_dir, exist_ok=True)
    class_info = {
        'classes': train_dataset.classes,
        'class_to_idx': train_dataset.class_to_idx
    }
    with open(f'{save_dir}/class_info.json', 'w') as f:
        json.dump(class_info, f, indent=2)
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # 모델 생성
    num_classes = len(train_dataset.classes)
    model = create_model(num_classes, model_name, pretrained=True)
    model = model.to(device)
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 학습 히스토리
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    # 학습 루프
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Scheduler step
        scheduler.step()
        
        # 히스토리 저장
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 결과 출력
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Best model 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'model_name': model_name,
                'num_classes': num_classes
            }
            torch.save(checkpoint, f'{save_dir}/best_model.pth')
            print(f"  ✓ Best model saved! (Val Acc: {val_acc:.2f}%)")
        
        # 주기적으로 체크포인트 저장
        if epoch % 10 == 0:
            torch.save(checkpoint, f'{save_dir}/checkpoint_epoch_{epoch}.pth')
    
    # 최종 히스토리 저장
    with open(f'{save_dir}/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"학습 완료!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    
    return model, history

# ============================================
# 7. 추론 함수 (학습된 모델 사용)
# ============================================
def load_trained_model(checkpoint_path, device='cuda'):
    """
    학습된 모델 로드
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 클래스 정보 로드
    checkpoint_dir = Path(checkpoint_path).parent
    with open(checkpoint_dir / 'class_info.json', 'r') as f:
        class_info = json.load(f)
    
    # 모델 생성 및 weight 로드
    model = create_model(
        checkpoint['num_classes'], 
        checkpoint['model_name'], 
        pretrained=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, class_info

def predict(model, image_path, class_info, img_size=224, device='cuda'):
    """
    단일 이미지 예측
    """
    transform = get_transforms(img_size, is_training=False)
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Top-5 결과
    top5_prob, top5_idx = torch.topk(probabilities, min(5, len(class_info['classes'])))
    
    results = []
    for prob, idx in zip(top5_prob, top5_idx):
        class_name = class_info['classes'][idx.item()]
        results.append((class_name, prob.item()))
    
    return results

# ============================================
# 사용 예시
# ============================================
if __name__ == "__main__":
    # 학습
    model, history = train_model(
        train_dir='data/train',
        val_dir='data/val',
        model_name='efficientnet_b3',
        num_epochs=50,
        batch_size=32,
        learning_rate=1e-4,
        img_size=224,
        save_dir='checkpoints'
    )
    
    # 추론
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, class_info = load_trained_model('checkpoints/best_model.pth', device)
    
    results = predict(model, 'test_image.jpg', class_info, device=device)
    
    print("\n예측 결과:")
    for class_name, prob in results:
        print(f"{class_name}: {prob*100:.2f}%")