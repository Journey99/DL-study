# 📚 Foundation Models 총정리

AI 연구와 산업 전반에서 *Foundation Model*은 핵심적인 기술 축으로 자리 잡고 있다.  
이 문서는 Foundation Model의 개념, 특징, 대표 모델, 구조적 발전 흐름, 핵심 개념 등을 정리한 자료이다.

---

## ✅ 1. Foundation Model이란?

**Foundation Model**은  
- 초거대 데이터로 사전 학습(pre-training)되고  
- 다양한 다운스트림(downstream) 작업에  
- 범용적으로 활용 가능한  
**초대규모 기반 모델**이다.

즉, 단일 목적 모델이 아니라  
여러 작업의 기반(foundation)이 되는 사전학습 모델이다.

---

## ✅ 2. Foundation Model의 핵심 특징

### ⭐ 2.1 초거대 데이터로 학습
- 이미지/텍스트 수억~수십억  
- 멀티모달 데이터 포함  
- 지도/비지도/자기지도 방식 혼합  

### ⭐ 2.2 범용성 (General-purpose)
하나의 모델로 다음 작업들을 수행할 수 있음:

- Image classification  
- Object detection  
- Segmentation  
- OCR  
- Captioning  
- VQA  
- Grounding  
- Text-to-image 제어  

### ⭐ 2.3 Few-shot / Zero-shot 능력
훈련 데이터가 거의 없어도 높은 성능

### ⭐ 2.4 Multimodal 확장
이미지 + 텍스트 + 오디오 등 다양한 modality 처리

### ⭐ 2.5 초대규모 파라미터  
수억 → 수십억 → 수백억 → 수조 파라미터까지 확장.

---

## ✅ 3. Foundation Model의 필요성

기존 딥러닝:  
**one task = one model**

Foundation 모델 시대:  
**one general model = many tasks**

즉, 하나의 모델이 다양한 태스크 수행.

---

## ✅ 4. 대표 Foundation Models

### 🔶 4.1 Vision Foundation Models

| 모델 | 발표연도 | 특징 |
|------|-----------|--------|
| CLIP (OpenAI) | 2021 | 이미지–텍스트 멀티모달 contrastive pretraining |
| DINO / DINOv2 (Meta) | 2021 / 2023 | self-supervised representation learning |
| SAM (Meta) | 2023 | prompt-based segmentation |
| Florence-2 (Microsoft) | 2024 | 범용 vision-language multitask 모델 |
| InternImage / InternViT | 2023~ | 고성능 ViT 기반 모델 |

---

### 🔶 4.2 Language Foundation Models

| 모델 | 발표연도 | 특징 |
|------|-----------|--------|
| GPT-3 / GPT-4 / GPT-5 | 2020~2025 | LLM, 범용 reasoning |
| PaLM / Gemini | 2022~2024 | Google 초거대 LLM |
| LLaMA 시리즈 | 2023~ | 오픈소스 고성능 LLM |

---

### 🔶 4.3 Multimodal Foundation Models

| 모델 | 발표연도 | 특징 |
|------|-----------|--------|
| CLIP | 2021 | 이미지–텍스트 alignment |
| BLIP / BLIP-2 | 2022~2023 | captioning/VQA |
| Flamingo | 2022 | few-shot multimodal reasoning |
| Grounding DINO | 2023 | detection + grounding |
| Kosmos / LLaVA | 2023~ | 이미지 이해 + 언어 모델 |
| GPT-4o / Omni Models | 2024~ | 이미지·텍스트·오디오 통합 reasoning |

---

## ✅ 5. Vision Foundation Model 타임라인

| 시대 | 주요 모델 | 특징 |
|------|-----------|---------|
| 2020–2021 | ViT, DINO | Self-supervised visual representation |
| 2021–2022 | CLIP | 멀티모달 contrastive learning |
| 2023 | SAM | Segment Anything 기반 segmentation 혁신 |
| 2023–2024 | DINOv2, Grounding DINO | 고성능 representation + detection |
| 2024–2025 | Florence-2, GPT-Vision 계열 | 통합 멀티모달 reasoning |

---

## ✅ 6. Foundation Model의 주요 학습 방식

### ✅ 6.1 Self-Supervised Learning  
라벨 없이 데이터의 패턴을 학습.  
예: DINO, MAE, SimCLR  

### ✅ 6.2 Contrastive Learning  
positive pair·negative pair 간 거리 차이를 학습.  
예: CLIP, ALIGN  

### ✅ 6.3 Distillation  
큰 모델의 지식을 작은 모델이 학습.  
예:  
- DINO → self-distillation  
- BLIP-2 → Q-Former가 LLM에서 지식 추출  

---

## ✅ 7. 장점 & 한계

### ✅ 장점
- 범용성 매우 높음  
- Zero-shot · Few-shot 가능  
- 다운스트림 학습 비용 감소  
- 멀티모달 확장 용이  
- 실무 적용 쉬움  

### ✅ 한계
- 막대한 학습 비용  
- 추론 비용 증가  
- 데이터 편향 문제  
- 모델 해석 어려움  

---

## ✅ 8. Foundation Model이 만든 변화

| 과거 | 현재 |
|------|--------|
| 태스크마다 모델 따로 학습 | Foundation Model 하나로 통합 |
| Supervised 데이터 필요 | Self-supervised 대규모 학습 |
| Vision / NLP 분리 | 멀티모달 통합 |
| 실무 적용 난이도 높음 | Prompt · 간단한 fine-tuning |

---

## ✅ 9. 참고 링크

- https://openai.com/research/clip  
- https://ai.meta.com/research/publications/dinov2  
- https://segment-anything.com  
- https://arxiv.org/abs/2301.12597  
- https://github.com/microsoft/Florence-2  

