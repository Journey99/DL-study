import torch
import clip
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def load_clip_model(model_name='ViT-B/32', device=None):
    """
    CLIP 모델 로드
    
    Available models:
        - 'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64'
        - 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'
    
    Args:
        model_name: 사용할 모델 이름
        device: cuda 또는 cpu
    
    Returns:
        model, preprocess, device
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading CLIP model: {model_name}")
    print(f"Device: {device}")
    
    model, preprocess = clip.load(model_name, device=device)
    
    print(f"✓ Model loaded successfully")
    print(f"  - Input resolution: {model.visual.input_resolution}")
    print(f"  - Context length: {model.context_length}")
    print(f"  - Vocab size: {model.vocab_size}")
    
    return model, preprocess, device


def encode_image(model, preprocess, image_path, device):
    """
    이미지를 CLIP embedding으로 변환
    
    Args:
        model: CLIP 모델
        preprocess: CLIP 전처리 함수
        image_path: 이미지 파일 경로
        device: cuda/cpu
    
    Returns:
        image_features: normalized embedding vector (512-dim for ViT-B/32)
    """
    # 이미지 로드 및 전처리
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    
    # 이미지 인코딩
    with torch.no_grad():
        image_features = model.encode_image(image)
        # L2 normalization (cosine similarity 계산용)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    return image_features


def encode_text(model, texts, device):
    """
    텍스트를 CLIP embedding으로 변환
    
    Args:
        model: CLIP 모델
        texts: 텍스트 리스트 (str 또는 list of str)
        device: cuda/cpu
    
    Returns:
        text_features: normalized embedding vectors
    """
    # 단일 텍스트를 리스트로 변환
    if isinstance(texts, str):
        texts = [texts]
    
    # 텍스트 토크나이징
    text_tokens = clip.tokenize(texts).to(device)
    
    # 텍스트 인코딩
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        # L2 normalization
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    return text_features


# 유사도 계산
def compute_similarity(image_features, text_features, temperature=100.0):
    """
    이미지-텍스트 유사도 계산
    
    Args:
        image_features: 이미지 embedding (N, D)
        text_features: 텍스트 embedding (M, D)
        temperature: CLIP의 학습된 temperature parameter (보통 100)
    
    Returns:
        similarity: cosine similarity matrix (N, M)
        probabilities: softmax probabilities (N, M)
    """
    # Cosine similarity (이미 normalized되어 있으므로 dot product)
    similarity = temperature * image_features @ text_features.T
    
    # Softmax로 확률 변환
    probabilities = similarity.softmax(dim=-1)
    
    return similarity, probabilities


def basic_inference_example(model, preprocess, device):
    """
    이미지 1개 + 여러 텍스트 설명으로 기본 추론
    """
    print("\n" + "="*60)
    print("기본 추론 예시: 이미지-텍스트 매칭")
    print("="*60)
    
    # 테스트 이미지
    image_path = "test_image.jpg"  # 본인의 이미지 경로로 변경
    
    # 후보 텍스트들
    texts = [
        "a photo of a cat",
        "a photo of a dog",
        "a photo of a bird",
        "a photo of a car",
        "a photo of a person"
    ]
    
    # 인코딩
    image_features = encode_image(model, preprocess, image_path, device)
    text_features = encode_text(model, texts, device)
    
    # 유사도 계산
    similarity, probabilities = compute_similarity(image_features, text_features)
    
    # 결과 출력
    print(f"\n이미지: {image_path}")
    print("\n매칭 결과:")
    for text, prob in zip(texts, probabilities[0]):
        print(f"  {text:30s} {prob.item()*100:6.2f}%")
    
    # 가장 높은 확률
    best_idx = probabilities[0].argmax().item()
    print(f"\n✓ 최고 매칭: {texts[best_idx]} ({probabilities[0][best_idx].item()*100:.2f}%)")
    
    return similarity, probabilities


def batch_image_inference(model, preprocess, image_paths, text_query, device):
    """
    여러 이미지에 대해 하나의 텍스트 쿼리로 검색
    
    Args:
        image_paths: 이미지 경로 리스트
        text_query: 검색할 텍스트 (str)
    
    Returns:
        scores: 각 이미지의 유사도 점수
    """
    print("\n" + "="*60)
    print(f"배치 검색: '{text_query}'")
    print("="*60)
    
    # 모든 이미지 인코딩
    image_features_list = []
    for img_path in image_paths:
        img_feat = encode_image(model, preprocess, img_path, device)
        image_features_list.append(img_feat)
    
    image_features = torch.cat(image_features_list, dim=0)
    
    # 텍스트 인코딩
    text_features = encode_text(model, text_query, device)
    
    # 유사도 계산
    similarity, probabilities = compute_similarity(image_features, text_features)
    
    # 결과 정렬
    scores = similarity[:, 0].cpu().numpy()
    sorted_indices = np.argsort(scores)[::-1]
    
    print("\n검색 결과 (유사도 높은 순):")
    for rank, idx in enumerate(sorted_indices, 1):
        print(f"  {rank}. {image_paths[idx]:30s} score: {scores[idx]:.4f}")
    
    return scores


def analyze_embeddings(model, preprocess, device):
    """
    CLIP 임베딩의 특성 분석
    """
    print("\n" + "="*60)
    print("임베딩 분석")
    print("="*60)
    
    # 샘플 이미지와 텍스트
    image_path = "test_image.jpg"
    text = "a photo of a cat"
    
    # 인코딩
    image_features = encode_image(model, preprocess, image_path, device)
    text_features = encode_text(model, text, device)
    
    # 임베딩 정보
    print(f"\n이미지 임베딩:")
    print(f"  - Shape: {image_features.shape}")
    print(f"  - Norm: {image_features.norm().item():.4f}")
    print(f"  - Mean: {image_features.mean().item():.4f}")
    print(f"  - Std: {image_features.std().item():.4f}")
    
    print(f"\n텍스트 임베딩:")
    print(f"  - Shape: {text_features.shape}")
    print(f"  - Norm: {text_features.norm().item():.4f}")
    print(f"  - Mean: {text_features.mean().item():.4f}")
    print(f"  - Std: {text_features.std().item():.4f}")
    
    # Cosine similarity
    cosine_sim = (image_features @ text_features.T).item()
    print(f"\nCosine Similarity: {cosine_sim:.4f}")
    
    return image_features, text_features

def text_variation_experiment(model, preprocess, image_path, device):
    """
    동일한 개념을 다른 방식으로 표현했을 때 유사도 비교
    """
    print("\n" + "="*60)
    print("텍스트 표현 변화 실험")
    print("="*60)
    
    # 같은 개념, 다른 표현
    text_variations = [
        "cat",
        "a cat",
        "a photo of a cat",
        "a picture of a cat",
        "an image of a cat",
        "a cute cat",
        "a cat sitting",
        "feline animal"
    ]
    
    # 이미지 인코딩
    image_features = encode_image(model, preprocess, image_path, device)
    
    print(f"\n이미지: {image_path}")
    print("\n다양한 텍스트 표현의 유사도:")
    
    for text in text_variations:
        text_features = encode_text(model, text, device)
        similarity = (image_features @ text_features.T).item()
        print(f"  '{text:30s}' → {similarity:.4f}")


if __name__ == "__main__":
    # 모델 로드
    model, preprocess, device = load_clip_model('ViT-B/32')
    
    # 1. 기본 추론
    # basic_inference_example(model, preprocess, device)
    
    # 2. 배치 검색
    # image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
    # batch_image_inference(model, preprocess, image_paths, "a cat", device)
    
    # 3. 임베딩 분석
    # analyze_embeddings(model, preprocess, device)
    
    # 4. 텍스트 표현 실험
    # text_variation_experiment(model, preprocess, "cat.jpg", device)
    
    print("\n" + "="*60)
    print("CLIP 기본 추론 완료!")
    print("="*60)
    print("\n사용 가능한 함수들:")
    print("  - encode_image(): 이미지 → 임베딩")
    print("  - encode_text(): 텍스트 → 임베딩")
    print("  - compute_similarity(): 유사도 계산")
    print("  - basic_inference_example(): 기본 매칭")
    print("  - batch_image_inference(): 배치 검색")
    print("  - analyze_embeddings(): 임베딩 분석")
    print("  - text_variation_experiment(): 텍스트 표현 비교")