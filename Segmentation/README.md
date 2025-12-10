# 📚 이미지 분할(Segmentation) 대표 모델 총정리

이 문서는 이미지 분할(Semantic / Instance / Panoptic / Interactive) 분야의 주요 모델들을  
**타임라인 → 모델 비교 → 코드 사용법 → 주요 개념 → 실무 선택 가이드** 순으로 정리한 README입니다.

---

# 🧭 1) Segmentation 모델 타임라인 요약

| 시대 | 주요 모델 | 비고 |
|------|-----------|------|
| 2015 | **U-Net** | Encoder–Decoder + Skip Connection의 시작 (Medical → 일반 segmentation로 확장) |
| 2017 | **PSPNet**, **Mask R-CNN** | Pyramid Pooling / Instance Segmentation 전성기 |
| 2018 | **DeepLabv3+** | ASPP + Encoder–Decoder, 경계 정교화 |
| 2019 | **HRNet** | High-Resolution 유지로 정확도 향상 |
| 2021 | **SegFormer** | Transformer 기반, 효율적인 semantic segmentation |
| 2022 | **Mask2Former** | Semantic / Instance / Panoptic 통합 아키텍처 |
| 2023 | **SAM (Segment Anything)** | Prompt 기반 제로샷·인터랙티브 모델 시대 개막 |

---

# 📊 2) 대표 모델 비교표

| 모델 | 발표연도 | 태스크 | 구조 특징 | 장점 | 한계 |
|------|---------:|--------|-----------|------|------|
| **U-Net** | 2015 | Semantic | Encoder–Decoder + Skip Connection | 적은 데이터도 잘 학습, 간단 | 자연 이미지에서 한계 |
| **PSPNet** | 2017 | Semantic | Pyramid Pooling Module | 강한 global context | 연산량 증가 |
| **Mask R-CNN** | 2017 | Instance | Faster R-CNN + Mask Head + RoIAlign | Instance SOTA, 안정적 | 무거움, 느림 |
| **DeepLabv3+** | 2018 | Semantic | ASPP + Depthwise Separable + Decoder | 경계 정밀도 ↑ | 구조 다소 복잡 |
| **HRNet** | 2019 | Semantic/Pose | Multi-resolution 병렬 유지 | 정교한 경계 처리 | 메모리 많이 듦 |
| **SegFormer** | 2021 | Semantic | Hierarchical Transformer Encoder + MLP Decoder | 빠르고 가벼움, 정확도 높음 | 아주 미세한 경계는 약함 |
| **Mask2Former** | 2022 | Semantic/Instance/Panoptic | Mask-Transformer + Masked Attention | 모든 분할 태스크 통합 | 학습 복잡도 있음 |
| **SAM** | 2023 | Interactive/Zero-shot | Promptable Segmentation + 대규모 pretrain | 제로샷 강력, 라벨링 비용 ↓ | 특정 도메인 fine-tuning 필요 |

---


# 🔧 3) 모델별 코드 사용법 (표 + 코드 스니펫)

| 모델 | 라이브러리 | 사용 방법 |
|------|------------|-----------|
| **U-Net** | `segmentation_models_pytorch` | ```python\nimport segmentation_models_pytorch as smp\nmodel = smp.Unet('resnet34', pretrained=True)\n``` |
| **PSPNet** | `segmentation_models_pytorch` | ```python\nmodel = smp.PSPNet(encoder_name='resnet50', pretrained=True)\n``` |
| **DeepLabv3+** | `torchvision` | ```python\nfrom torchvision.models.segmentation import deeplabv3_resnet50\nmodel = deeplabv3_resnet50(pretrained=True)\n``` |
| **Mask R-CNN** (Instance) | `detectron2` | ```python\nfrom detectron2.engine import DefaultPredictor\n``` |
| **SegFormer** | `mmsegmentation` | ```bash\npython tools/train.py configs/segformer/segformer_mit-b0_512x512.py\n``` |
| **Mask2Former** | `mmsegmentation` | ```bash\npython tools/train.py configs/mask2former/mask2former_r50.py\n``` |
| **SAM** | `segment-anything` | ```python\nfrom segment_anything import SamPredictor\n``` |

> 참고  
- Semantic 위주: `torchvision`, `smp`  
- Instance/Panoptic: `detectron2`, `mmsegmentation`  
- Foundation Models: `SAM`  


---


# 🔎 4) 주요 개념 요약

## ✅ Semantic / Instance / Panoptic
- **Semantic segmentation**: 각 픽셀의 클래스를 예측 (개별 object 구분 X)
- **Instance segmentation**: 같은 클래스라도 개체별로 mask 구분
- **Panoptic segmentation**: semantic + instance를 통합하여 모든 픽셀에 “클래스 + 인스턴스 ID”



## ✅ Encoder–Decoder 구조 (U-Net 등)
- Encoder: Downsampling, high-level feature 추출  
- Decoder: Upsampling, 공간 해상도 복원  
- Skip connection: 로우레벨 정보 전달 → 경계 품질 크게 향상



## ✅ Atrous/Dilated Convolution & ASPP (DeepLab 시리즈)
- Receptive field를 늘리면서 해상도 유지  
- 다양한 dilation rate를 병렬 적용 → Multi-scale context 학습


## ✅ ROIAlign & Instance Masking (Mask R-CNN)
- Feature map에서 정밀하게 ROI를 추출하는 연산  
- Instance-level mask head로 개체 구분


## ✅ Transformer 기반 분할 (SegFormer, Mask2Former)
- CNN의 국소성 한계를 넘어 **전역 문맥을 효율적으로 학습**  
- Mask2Former는 semantic/instance/panoptic을 모두 “mask query + masked-attention”으로 통합

---

## ✅ Promptable Segmentation (SAM)
- Point / Box / Mask prompt로 특정 영역 강조  
- 대규모 pretrain된 embedding + prompt encoder 사용  
- 라벨링 자동화, 제로샷 inference 강력


# 🧭 5) 실무에서 사용하는 모델은?

### ✅ Task별 추천 모델

| Task | 추천 모델 | 이유 |
|------|-----------|------|
| **의료 · 위성 · 라벨 적은 도메인** | U-Net, U-Net++ | 적은 데이터에서 강함 |
| **일반 Semantic Segmentation** | DeepLabv3+, SegFormer | 성능 + 속도 균형 |
| **Instance Segmentation** | Mask R-CNN | 견고함 + 풍부한 구현체 |
| **Panoptic Segmentation** | Mask2Former | 통합 모델 구조 |
| **제로샷/라벨링 자동화** | SAM | prompt 기반 + 대규모 사전학습 |

---

# ✅ 6) 최종 요약

- ✅ 2015~2019: CNN 기반(U-Net, DeepLab, PSPNet, HRNet)  
- ✅ 2021~2022: Transformer 기반(SegFormer, Mask2Former) → 정확도·범용성 증가  
- ✅ 2023~: Foundation 모델 시대 (SAM) → 제로샷 & 인터랙티브 세그멘테이션  

오늘날 실무에서 가장 많이 쓰이는 모델:  
**DeepLabv3+, U-Net 계열, SegFormer, Mask2Former, SAM**

---

# 📚 7) 참고 자료 (논문/Repo)

- U-Net (2015): https://arxiv.org/abs/1505.04597  
- PSPNet (2017): https://arxiv.org/abs/1612.01105  
- Mask R-CNN (2017): https://arxiv.org/abs/1703.06870  
- DeepLabv3+ (2018): https://arxiv.org/abs/1802.02611  
- HRNet (2019): https://arxiv.org/abs/1908.07919  
- SegFormer (2021): https://arxiv.org/abs/2105.15203  
- Mask2Former (2022): https://arxiv.org/abs/2112.01527  
- Segment Anything (2023): https://arxiv.org/abs/2304.02643  


# ✅ 8) instance segmentation 가능여부
| 모델 | 원래 용도 | Instance 가능? | 방법 |
|:-----|:---------|:--------------|:-----|
| **U-Net** | Semantic Segmentation | ✅ 가능 | Instance head 추가 (Center/Offset/Gradient) |
| **DeepLabv3+** | Semantic Segmentation | ✅ 가능 | Instance head 추가 or Two-stage 방식 |
| **SegFormer** | Semantic Segmentation | ✅ 가능 | Instance head 추가 or Post-processing |
| **Mask R-CNN** | Instance Segmentation | ✅ 원래 가능 | Two-stage design (RPN + ROI Head) |
| **Mask2Former** | Universal Segmentation | ✅ 원래 가능 | Query-based unified architecture |

### 원래부터 Instance 가능한 모델

| 모델 | 방식 | 장점 | 단점 |
|:-----|:-----|:-----|:-----|
| **Mask R-CNN** | Two-stage (Proposal-based) | 높은 정확도, 안정적 | 느림 |
| **Cascade Mask R-CNN** | Multi-stage refinement | 매우 높은 정확도 | 매우 느림 |
| **YOLACT** | One-stage (Prototype-based) | 빠름 (30+ FPS) | 정확도 낮음 |
| **SOLOv2** | One-stage (Location-based) | 빠르고 정확 | 작은 객체 약함 |
| **Mask2Former** | Query-based Transformer | SOTA, Universal | 무거움 |
| **OneFormer** | Task-conditional | 하나로 모든 task | 매우 무거움 |

## 도메인별 추천 모델

### 의료 영상 (세포, 핵, 장기)

| 용도 | 추천 모델 | 이유 |
|:-----|:---------|:-----|
| **세포 Instance** | U-Net + Gradient (Cellpose) | Round object, touching 처리 우수 |
| **핵 Instance** | U-Net + HV maps (Hover-Net) | 병리 이미지 특화 |
| **장기 Semantic** | U-Net, nnU-Net | 검증된 성능 |
| **작은 데이터셋** | U-Net + Watershed | 간단하고 효과적 |

### 자율주행

| 용도 | 추천 모델 | 이유 |
|:-----|:---------|:-----|
| **Scene Understanding** | Mask2Former (Panoptic) | Semantic + Instance 통합 |
| **실시간 처리** | YOLACT or SegFormer-B0 | 속도 중요 |
| **최고 정확도** | Cascade Mask R-CNN | 안전 critical |

### 일반 객체 인식

| 용도 | 추천 모델 | 이유 |
|:-----|:---------|:-----|
| **연구/프로토타입** | Mask R-CNN | 범용성, 안정성 |
| **프로덕션** | SOLOv2 or Mask2Former | 속도와 정확도 균형 |
| **실시간 요구** | YOLACT | 30+ FPS |
| **Semantic만 필요** | SegFormer | 가볍고 빠름 |