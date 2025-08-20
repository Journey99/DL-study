
# 📚 이미지 분류(Image Classification) 대표 모델 총정리

이 문서는 이미지 분류에 사용되는 대표적인 모델들의 발전 흐름과 구조, 성능, 사용 가능한 코드/라이브러리 정보를 포함한 정리입니다.



## 🧭 이미지 분류 모델 타임라인 요약

| 시대 | 주요 모델 | 비고 |
|------|-----------|------|
| 2012~2014 | AlexNet, VGG | CNN의 부흥기 |
| 2015~2017 | ResNet, Inception, DenseNet | 깊은 네트워크 + skip 연결 |
| 2018~2020 | EfficientNet, RegNet | 자동 구조 탐색 (NAS) |
| 2021~현재 | ViT, Swin Transformer, ConvNeXt | Transformer 기반 또는 CNN-Transformer 하이브리드 |

---

## 📊 대표 모델 비교표

| 모델명 | 발표연도 | 구조 특징 | 장점 | 한계 | Params (M) | Top-1@ImageNet |
|--------|----------|------------|------|------|-------------|----------------|
| **AlexNet** | 2012 | 5 Conv + 3 FC | ReLU, GPU 학습 도입 | 깊이 부족 | 60M | 57.0% |
| **VGG16** | 2014 | 3×3 Conv 반복 | 단순한 구조, 직관적 | 파라미터 많음 | 138M | 71.5% |
| **GoogLeNet (Inception v1)** | 2014 | Inception module | 계산 효율 ↑ | 구조 복잡 | 7M | 69.8% |
| **ResNet-50** | 2015 | Residual block | 깊은 네트워크 안정화 | 연산량 많음 | 25M | 76.2% |
| **DenseNet-121** | 2017 | Dense connection | 파라미터 효율 ↑ | 병렬화 어려움 | 8M | 75.0% |
| **EfficientNet-B0** | 2019 | Compound scaling | 높은 정확도, 적은 연산 | 구조 생성 복잡 | 5M | 77.1% |
| **RegNetY-16GF** | 2020 | 구조 패턴 수식화 | 최적화 자동화 | 직관성 낮음 | 84M | 80.9% |
| **ViT-B/16** | 2021 | Patch + Transformer | 전역 정보 처리 강함 | 많은 데이터 필요 | 86M | 77.9% |
| **DeiT-B** | 2021 | ViT + Distillation | 소량 데이터도 학습 | ViT급 성능 | 86M | 81.8% |
| **Swin-T** | 2021 | Shifted Windows | 연산 효율, 계층 구조 | 구조 다소 복잡 | 29M | 81.3% |
| **ConvNeXt-T** | 2022 | ViT 스타일 CNN | ViT 성능 + CNN 친화성 | 다소 무거움 | 29M | 82.1% |

---

## 🔧 모델별 코드 사용법

| 모델 | 라이브러리 | 사용 방법 |
|------|------------|-----------|
| **AlexNet** | `torchvision` | `torchvision.models.alexnet(pretrained=True)` |
| **VGG16** | `torchvision` | `torchvision.models.vgg16(pretrained=True)` |
| **GoogLeNet** | `torchvision` | `torchvision.models.googlenet(pretrained=True)` |
| **ResNet-50** | `torchvision` | `torchvision.models.resnet50(pretrained=True)` |
| **DenseNet-121** | `torchvision` | `torchvision.models.densenet121(pretrained=True)` |
| **EfficientNet-B0** | `torchvision` | `torchvision.models.efficientnet_b0(pretrained=True)` |
| **RegNetY-16GF** | `torchvision` | `torchvision.models.regnet_y_16gf(pretrained=True)` |
| **ViT-B/16** | `transformers` by HuggingFace | `AutoModel.from_pretrained("google/vit-base-patch16-224")` |
| **DeiT-B** | `transformers` or `timm` | `timm.create_model("deit_base_patch16_224", pretrained=True)` |
| **Swin-T** | `timm` | `timm.create_model("swin_tiny_patch4_window7_224", pretrained=True)` |
| **ConvNeXt-T** | `timm` | `timm.create_model("convnext_tiny", pretrained=True)` |

---

## 🔍 주요 개념 요약

| 개념 | 설명 |
|------|------|
| **Residual connection** | 깊은 네트워크 학습 안정화 (ResNet) |
| **Inception module** | 다양한 커널 병렬 사용 (GoogLeNet) |
| **Dense connection** | 모든 레이어를 연결 (DenseNet) |
| **Neural Architecture Search (NAS)** | 구조를 자동 탐색 (EfficientNet) |
| **Transformer** | 전역 attention 사용 (ViT, Swin) |
| **Window Attention** | 국소 attention + 전역 흐름 (Swin) |

---

## 🔚 요약 정리

- **CNN 계열**: VGG → ResNet → DenseNet → EfficientNet
- **Transformer 계열**: ViT → Swin → ConvNeXt
- **추세**: ViT 계열이 급성장 중이지만, CNN도 계속 진화 (ConvNeXt 등)

---



## ✅ 참고

- PyTorch Docs: https://pytorch.org/vision/stable/models.html
- HuggingFace ViT: https://huggingface.co/google/vit-base-patch16-224
- Timm 모델 리스트: https://rwightman.github.io/pytorch-image-models/

