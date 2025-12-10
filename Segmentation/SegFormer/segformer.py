"""
SegFormer 사용 예제 코드
"""

# ===== 1. Hugging Face로 사용 (가장 간단) =====
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO

def segformer_huggingface_inference(image_path):
    """Hugging Face SegFormer 사용"""
    
    # 모델 및 프로세서 로드
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
    
    # 이미지 로드
    image = Image.open(image_path).convert('RGB')
    
    # 전처리
    inputs = processor(images=image, return_tensors="pt")
    
    # 추론
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # (batch_size, num_labels, height, width)
    
    # Upsample to original size
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],  # (height, width)
        mode="bilinear",
        align_corners=False
    )
    
    # Get prediction
    pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
    
    return pred_seg, image


# ===== 2. Segmentation Models Pytorch로 사용 =====
import segmentation_models_pytorch as smp
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

def segformer_smp_inference(image_path, num_classes=150):
    """SMP SegFormer 사용"""
    
    # 모델 생성
    model = smp.SegFormer(
        encoder_name="mit_b5",           # mit_b0, mit_b1, ..., mit_b5
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    )
    
    # Pretrained weight 로드 (optional)
    # checkpoint = torch.load('segformer_b5_ade20k.pth')
    # model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # 이미지 로드 및 전처리
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = A.Compose([
        A.Resize(640, 640),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    transformed = transform(image=image)
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # 추론
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = output.argmax(dim=1)[0].cpu().numpy()
    
    return pred_mask, image


# ===== 3. Custom Training with SMP =====
from torch.utils.data import DataLoader

def train_segformer_custom():
    """Custom dataset으로 SegFormer 학습"""
    
    # 모델 생성
    model = smp.SegFormer(
        encoder_name="mit_b3",  # 효율성과 성능의 균형
        encoder_weights="imagenet",
        in_channels=3,
        classes=5,  # 본인 데이터셋 클래스 수
        activation=None,
    )
    
    # Loss & Optimizer
    criterion = smp.losses.DiceLoss(mode='multiclass')
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer, 
        total_iters=50, 
        power=0.9
    )
    
    # Training loop (간략화)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # train_loader = DataLoader(...)  # 본인 dataset
    # for epoch in range(50):
    #     for images, masks in train_loader:
    #         images, masks = images.to(device), masks.to(device)
    #         outputs = model(images)
    #         loss = criterion(outputs, masks)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     scheduler.step()
    
    return model


# ===== 4. 시각화 함수 =====
def visualize_segmentation(image, seg_mask, num_classes=150):
    """세그멘테이션 결과 시각화"""
    
    # 컬러 팔레트 생성
    palette = np.random.randint(0, 255, (num_classes, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0]  # background
    
    # 마스크를 컬러로 변환
    color_seg = palette[seg_mask]
    
    # 원본 이미지와 오버레이
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # 크기 맞추기
    if color_seg.shape[:2] != image.shape[:2]:
        color_seg = cv2.resize(color_seg, (image.shape[1], image.shape[0]))
    
    overlay = (0.6 * image + 0.4 * color_seg).astype(np.uint8)
    
    # 플롯
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(color_seg)
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('segformer_result.png', dpi=150, bbox_inches='tight')
    plt.show()


# ===== 5. 다양한 SegFormer 모델 비교 =====
def compare_segformer_variants():
    """SegFormer B0~B5 모델 정보"""
    
    models_info = {
        'B0': {
            'encoder': 'mit_b0',
            'params': '3.8M',
            'gflops': 8.4,
            'ade20k_miou': 37.4,
            'use_case': 'Mobile/Edge devices'
        },
        'B1': {
            'encoder': 'mit_b1',
            'params': '13.7M',
            'gflops': 16.0,
            'ade20k_miou': 42.2,
            'use_case': 'Real-time applications'
        },
        'B2': {
            'encoder': 'mit_b2',
            'params': '27.4M',
            'gflops': 62.4,
            'ade20k_miou': 46.5,
            'use_case': 'Balanced performance'
        },
        'B3': {
            'encoder': 'mit_b3',
            'params': '47.3M',
            'gflops': 79.0,
            'ade20k_miou': 48.4,
            'use_case': 'High accuracy'
        },
        'B4': {
            'encoder': 'mit_b4',
            'params': '64.1M',
            'gflops': 95.7,
            'ade20k_miou': 49.3,
            'use_case': 'Very high accuracy'
        },
        'B5': {
            'encoder': 'mit_b5',
            'params': '84.7M',
            'gflops': 167.0,
            'ade20k_miou': 51.8,
            'use_case': 'SOTA performance'
        }
    }
    
    print("=== SegFormer Model Variants ===\n")
    for name, info in models_info.items():
        print(f"SegFormer-{name}:")
        print(f"  Encoder: {info['encoder']}")
        print(f"  Parameters: {info['params']}")
        print(f"  GFLOPs: {info['gflops']}")
        print(f"  ADE20K mIoU: {info['ade20k_miou']}")
        print(f"  Use Case: {info['use_case']}")
        print()


# ===== 6. Hugging Face 사용 가능한 Pretrained 모델들 =====
def list_available_models():
    """사용 가능한 pretrained 모델 목록"""
    
    models = [
        "nvidia/segformer-b0-finetuned-ade-512-512",
        "nvidia/segformer-b1-finetuned-ade-512-512",
        "nvidia/segformer-b2-finetuned-ade-512-512",
        "nvidia/segformer-b3-finetuned-ade-512-512",
        "nvidia/segformer-b4-finetuned-ade-512-512",
        "nvidia/segformer-b5-finetuned-ade-640-640",
        
        "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
    ]
    
    print("=== Available Pretrained Models ===\n")
    for model in models:
        print(f"- {model}")


# ===== 실행 예제 =====
if __name__ == "__main__":
    print("=== SegFormer 사용 예제 ===\n")
    
    # 1. 모델 비교
    compare_segformer_variants()
    
    # 2. 사용 가능한 모델 목록
    list_available_models()
    
    # 3. 추론 예제 (이미지 경로 필요)
    # image_path = "your_image.jpg"
    
    # Hugging Face 방식
    # pred_seg, orig_image = segformer_huggingface_inference(image_path)
    # visualize_segmentation(orig_image, pred_seg, num_classes=150)
    
    # SMP 방식
    # pred_mask, orig_image = segformer_smp_inference(image_path)
    # visualize_segmentation(orig_image, pred_mask)
    
    print("\n이미지 경로를 설정하고 주석을 해제하여 실행하세요!")


# ===== 빠른 테스트용 코드 =====
def quick_test():
    """인터넷 이미지로 빠른 테스트"""
    
    # 샘플 이미지 다운로드
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    
    # SegFormer inference
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False
    )
    
    pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
    
    # 시각화
    visualize_segmentation(image, pred_seg, num_classes=150)
    print(f"Segmentation shape: {pred_seg.shape}")
    print(f"Unique classes: {np.unique(pred_seg)}")

# quick_test()  # 주석 해제하면 바로 테스트 가능