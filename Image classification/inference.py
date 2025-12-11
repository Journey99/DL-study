'''
EfficientNet (cnn 계열)
- 모델 크기와 정확도 간의 균형을 맞추기 위해 compound scaling 기법 사용
- 적은 파라미터로도 높은 성능을 발휘    
- 다양한 변형 모델로 제공되어 다양한 요구에 대응 가능

ConvNeXt (cnn 계열)
- meta가 만든 cnn의 최종 진화형
- transformer 스타일의 아키텍처를 cnn으로 재해석
- convnext v2 는 성능과 효율성 모두에서 뛰어남

ViT (transformer 계열)
- 이미지 분류에 transformer를 적용한 최초의 모델
- 이미지 패치를 입력으로 받아 self-attention 메커니즘을 사용
- 대규모 데이터셋에서 뛰어난 성능을 발휘

swin transformer (transformer 계열)
- 계층적 구조와 지역적 self-attention 메커니즘을 도입
- 이미지 내의 지역적 특징을 효과적으로 캡처
- classification, 객체 검출 등 다양한 비전 작업에 활용

Dinov2 (transformer 계열)   
- self-supervised learning 기반의 비전 모델
- ViT를 백본으로 사용하며, 라벨 없이 학습
- 대규모 이미지 데이터셋에서 사전 학습되어 강력한 표현력 보유

'''

import torch
import timm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================
# 1. EfficientNet
# ============================================
def load_efficientnet():
    model = timm.create_model('efficientnet_b3', pretrained=True)
    model = model.to(device)
    model.eval()
    
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    
    return model, transform

# ============================================
# 2. ConvNeXt
# ============================================
def load_convnext():
    model = timm.create_model('convnext_base', pretrained=True)
    model = model.to(device)
    model.eval()
    
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    
    return model, transform

# ============================================
# 3. Vision Transformer (ViT)
# ============================================
def load_vit():
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model = model.to(device)
    model.eval()
    
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    
    return model, transform

# ============================================
# 4. Swin Transformer
# ============================================
def load_swin():
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
    model = model.to(device)
    model.eval()
    
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    
    return model, transform

# ============================================
# 5. DINOv2
# ============================================
def load_dinov2():
    model = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True)
    model = model.to(device)
    model.eval()
    
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    
    return model, transform

# ============================================
# 추론 함수
# ============================================
def inference(model, transform, image_path, top_k=5):
    """
    단일 이미지 추론
    
    Args:
        model: 추론할 모델
        transform: 전처리 transform
        image_path: 이미지 경로
        top_k: 상위 k개 클래스 반환
    
    Returns:
        top_k개의 (class_idx, probability) 튜플 리스트
    """
    # 이미지 로드 및 전처리
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # 추론
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Top-k 결과
    top_prob, top_idx = torch.topk(probabilities, top_k)
    
    results = []
    for prob, idx in zip(top_prob, top_idx):
        results.append((idx.item(), prob.item()))
    
    return results

# ============================================
# 배치 추론 함수
# ============================================
def batch_inference(model, transform, image_paths, batch_size=32):
    """
    배치 추론
    
    Args:
        model: 추론할 모델
        transform: 전처리 transform
        image_paths: 이미지 경로 리스트
        batch_size: 배치 크기
    
    Returns:
        각 이미지의 예측 결과 리스트
    """
    from torch.utils.data import Dataset, DataLoader
    
    class ImageDataset(Dataset):
        def __init__(self, image_paths, transform):
            self.image_paths = image_paths
            self.transform = transform
        
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            img = Image.open(self.image_paths[idx]).convert('RGB')
            return self.transform(img)
    
    dataset = ImageDataset(image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    
    all_predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            outputs = model(batch)
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
    
    return all_predictions

# ============================================
# Feature Extraction (Transfer Learning용)
# ============================================
def extract_features(model, transform, image_path):
    """
    마지막 분류 레이어 전의 feature 추출
    
    Returns:
        feature vector (numpy array)
    """
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # forward_features는 timm의 표준 메서드
        features = model.forward_features(img_tensor)
        
        # Global Average Pooling (모델에 따라 이미 적용되어 있을 수 있음)
        if len(features.shape) == 4:  # (B, C, H, W)
            features = features.mean(dim=[2, 3])
        elif len(features.shape) == 3:  # (B, N, C) - Transformer
            features = features[:, 0]  # CLS token
    
    return features.cpu().numpy()

# ============================================
# ImageNet 레이블 로딩
# ============================================
def load_imagenet_labels():
    """
    ImageNet 1000 클래스 레이블 로드
    
    Returns:
        클래스 이름 리스트 (인덱스 = 클래스 ID)
    """
    import json
    import urllib.request
    
    try:
        # 간단한 레이블 버전
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        with urllib.request.urlopen(url) as f:
            labels = json.load(f)
        return labels
    except:
        print("레이블 다운로드 실패. 인덱스만 표시합니다.")
        return [f"class_{i}" for i in range(1000)]


# ============================================
if __name__ == "__main__":
    # ImageNet 클래스 레이블 (간단 버전)
    # 실제로는 imagenet_classes.txt 파일을 로드하는 것이 좋음
    
    class_labels = load_imagenet_labels()
    
    image_path = "your_image.jpg"
    
    print("\n=== 1. EfficientNet-B3 ===")
    model, transform = load_efficientnet()
    results = inference(model, transform, image_path)
    for idx, prob in results:
        print(f"{class_labels[idx]} (Class {idx}): {prob*100:.2f}%")
    
    print("\n=== 2. ConvNeXt-Base ===")
    model, transform = load_convnext()
    results = inference(model, transform, image_path)
    for idx, prob in results:
        print(f"{class_labels[idx]} (Class {idx}): {prob*100:.2f}%")
    
    print("\n=== 3. ViT-Base ===")
    model, transform = load_vit()
    results = inference(model, transform, image_path)
    for idx, prob in results:
        print(f"{class_labels[idx]} (Class {idx}): {prob*100:.2f}%")
    
    print("\n=== 4. Swin-Base ===")
    model, transform = load_swin()
    results = inference(model, transform, image_path)
    for idx, prob in results:
        print(f"{class_labels[idx]} (Class {idx}): {prob*100:.2f}%")
    
    print("\n=== 5. DINOv2-Base ===")
    model, transform = load_dinov2()
    results = inference(model, transform, image_path)
    for idx, prob in results:
        print(f"{class_labels[idx]} (Class {idx}): {prob*100:.2f}%")
    
    # Feature extraction 예시
    print("\n=== Feature Extraction ===")
    features = extract_features(model, transform, image_path)
    print(f"Feature shape: {features.shape}")
