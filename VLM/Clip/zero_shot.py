import torch
import clip
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


def load_clip_model(model_name='ViT-B/32'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP: {model_name}")
    print(f"Device: {device}\n")
    
    model, preprocess = clip.load(model_name, device=device)
    return model, preprocess, device


def load_cifar10(preprocess, batch_size=100):
    """
    CIFAR-10 test set 로드
    
    Returns:
        dataloader, class_names
    """
    # CIFAR-10 클래스 이름
    cifar10_classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    # Test set 다운로드 (자동)
    dataset = CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=preprocess
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    print(f"CIFAR-10 Test Set:")
    print(f"  - Images: {len(dataset)}")
    print(f"  - Classes: {len(cifar10_classes)}")
    print(f"  - Class names: {cifar10_classes}\n")
    
    return dataloader, cifar10_classes


def get_prompt_templates():
    """
    CLIP 논문에서 사용한 prompt engineering 기법
    여러 template으로 ensemble
    """
    templates = [
        'a photo of a {}.',
        'a blurry photo of a {}.',
        'a low resolution photo of a {}.',
        'a photo of the {}.',
        'a photo of a small {}.',
        'a photo of a large {}.',
        'a photo of my {}.',
        'a bright photo of a {}.',
        'a dark photo of a {}.',
        'a photo of the clean {}.',
        'a photo of the dirty {}.',
    ]
    return templates


def encode_text_classifier(model, class_names, templates, device):
    """
    클래스별로 여러 prompt template을 사용해 텍스트 임베딩 생성
    
    Args:
        model: CLIP 모델
        class_names: 클래스 이름 리스트
        templates: prompt template 리스트
        device: cuda/cpu
    
    Returns:
        text_features: [num_classes, embedding_dim]
    """
    print("Encoding text prompts...")
    
    # 모든 class × template 조합 생성
    all_texts = []
    for class_name in class_names:
        texts = [template.format(class_name) for template in templates]
        all_texts.extend(texts)
    
    print(f"  - Total prompts: {len(all_texts)}")
    print(f"  - Example: '{all_texts[0]}'")
    
    # 배치 단위로 텍스트 인코딩 (메모리 절약)
    batch_size = 256
    text_features_list = []
    
    with torch.no_grad():
        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i:i+batch_size]
            text_tokens = clip.tokenize(batch_texts).to(device)
            batch_features = model.encode_text(text_tokens)
            batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
            text_features_list.append(batch_features)
    
    text_features = torch.cat(text_features_list, dim=0)
    
    # 각 클래스별로 template들의 평균 임베딩 계산 (ensemble)
    text_features = text_features.view(len(class_names), len(templates), -1)
    text_features = text_features.mean(dim=1)  # 평균
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # 재정규화
    
    print(f"  - Final shape: {text_features.shape}")
    print("✓ Text encoding complete\n")
    
    return text_features


# Zero-shot Prediction
@torch.no_grad()
def zero_shot_classify(model, dataloader, text_features, device):
    """
    Zero-shot classification 수행
    
    Returns:
        predictions: 예측 클래스
        labels: 실제 레이블
        probabilities: 각 클래스의 확률
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    print("Running zero-shot classification...")
    
    for images, labels in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        
        # 이미지 인코딩
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Cosine similarity
        similarity = 100.0 * image_features @ text_features.T
        probabilities = similarity.softmax(dim=-1)
        
        # 예측
        predictions = probabilities.argmax(dim=-1)
        
        all_predictions.append(predictions.cpu())
        all_labels.append(labels)
        all_probabilities.append(probabilities.cpu())
    
    # 결과 합치기
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    all_probabilities = torch.cat(all_probabilities)
    
    return all_predictions, all_labels, all_probabilities


def evaluate_results(predictions, labels, class_names):
    """
    정확도 및 클래스별 성능 계산
    """
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Overall accuracy
    correct = (predictions == labels).sum().item()
    total = len(labels)
    accuracy = 100.0 * correct / total
    
    print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    # 클래스별 accuracy
    print("\n" + "-"*60)
    print("Per-class Accuracy:")
    print("-"*60)
    
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    for pred, label in zip(predictions, labels):
        class_total[label.item()] += 1
        if pred == label:
            class_correct[label.item()] += 1
    
    class_accuracies = []
    for i, class_name in enumerate(class_names):
        acc = 100.0 * class_correct[i] / class_total[i]
        class_accuracies.append(acc)
        print(f"  {class_name:12s}: {acc:6.2f}% ({class_correct[i]}/{class_total[i]})")
    
    # 평균 클래스별 accuracy (macro-average)
    mean_acc = np.mean(class_accuracies)
    print(f"\n  Mean Accuracy: {mean_acc:.2f}%")
    
    return accuracy, class_accuracies


def plot_confusion_matrix(predictions, labels, class_names, save_path='confusion_matrix.png'):
    """
    Confusion matrix 그리기
    """
    from sklearn.metrics import confusion_matrix
    
    # Confusion matrix 계산
    cm = confusion_matrix(labels.numpy(), predictions.numpy())
    
    # 정규화 (각 행의 합 = 1)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 플롯
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Accuracy'}
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('CLIP Zero-shot Classification - Confusion Matrix (CIFAR-10)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved: {save_path}")
    
    return cm


def compute_topk_accuracy(probabilities, labels, k=3):
    """
    Top-k accuracy 계산
    """
    topk_preds = probabilities.topk(k, dim=1)[1]
    correct = 0
    
    for i, label in enumerate(labels):
        if label in topk_preds[i]:
            correct += 1
    
    topk_acc = 100.0 * correct / len(labels)
    return topk_acc


def analyze_misclassifications(predictions, labels, probabilities, class_names, top_n=10):
    """
    가장 확신있게 틀린 케이스들 분석
    """
    print("\n" + "="*60)
    print("MISCLASSIFICATION ANALYSIS")
    print("="*60)
    
    # 틀린 예측들
    wrong_indices = (predictions != labels).nonzero(as_tuple=True)[0]
    
    if len(wrong_indices) == 0:
        print("No misclassifications!")
        return
    
    # 예측 확률이 높은 순으로 정렬
    wrong_probs = probabilities[wrong_indices]
    max_probs, _ = wrong_probs.max(dim=1)
    sorted_indices = max_probs.argsort(descending=True)[:top_n]
    
    print(f"\nTop {top_n} most confident mistakes:\n")
    
    for rank, idx in enumerate(sorted_indices, 1):
        original_idx = wrong_indices[idx].item()
        true_label = labels[original_idx].item()
        pred_label = predictions[original_idx].item()
        confidence = max_probs[idx].item()
        
        print(f"{rank}. Image #{original_idx}")
        print(f"   True: {class_names[true_label]:12s}")
        print(f"   Pred: {class_names[pred_label]:12s} (confidence: {confidence*100:.2f}%)")
        print()


def compare_prompt_templates(model, dataloader, class_names, device):
    """
    다양한 prompt template의 영향 비교
    """
    print("\n" + "="*60)
    print("PROMPT TEMPLATE COMPARISON")
    print("="*60 + "\n")
    
    # 다양한 template 조합 테스트
    template_sets = {
        'Simple': ['{}'],
        'Basic': ['a photo of a {}.'],
        'Detailed': [
            'a photo of a {}.',
            'a blurry photo of a {}.',
            'a low resolution photo of a {}.',
        ],
        'Full Ensemble': get_prompt_templates()
    }
    
    results = {}
    
    for name, templates in template_sets.items():
        print(f"Testing: {name} ({len(templates)} templates)")
        
        # 텍스트 인코딩
        text_features = encode_text_classifier(model, class_names, templates, device)
        
        # 평가
        predictions, labels, probabilities = zero_shot_classify(
            model, dataloader, text_features, device
        )
        
        # 정확도 계산
        accuracy = 100.0 * (predictions == labels).sum().item() / len(labels)
        results[name] = accuracy
        
        print(f"  → Accuracy: {accuracy:.2f}%\n")
    
    # 결과 요약
    print("="*60)
    print("SUMMARY")
    print("="*60)
    for name, acc in results.items():
        print(f"{name:20s}: {acc:.2f}%")
    
    return results

def main():
    # 모델 로드
    model, preprocess, device = load_clip_model('ViT-B/32')
    
    # 데이터 로드
    dataloader, class_names = load_cifar10(preprocess, batch_size=100)
    
    # Prompt templates
    templates = get_prompt_templates()
    
    # 텍스트 인코딩
    text_features = encode_text_classifier(model, class_names, templates, device)
    
    # Zero-shot classification
    predictions, labels, probabilities = zero_shot_classify(
        model, dataloader, text_features, device
    )
    
    # 평가
    accuracy, class_accuracies = evaluate_results(predictions, labels, class_names)
    
    # Top-3 accuracy
    top3_acc = compute_topk_accuracy(probabilities, labels, k=3)
    print(f"\nTop-3 Accuracy: {top3_acc:.2f}%")
    
    # Confusion matrix
    plot_confusion_matrix(predictions, labels, class_names)
    
    # 오분류 분석
    analyze_misclassifications(predictions, labels, probabilities, class_names)
    
    # Prompt template 비교
    # compare_prompt_templates(model, dataloader, class_names, device)
    
    print("\n" + "="*60)
    print("Zero-shot Classification Complete!")
    print("="*60)

if __name__ == "__main__":
    main()