# Multimodal AI 기초

## 1. Unimodal vs Multimodal

### Unimodal AI (단일 모달리티)
하나의 데이터 타입만 처리:

- Vision-only: 이미지 분류, 객체 탐지 (ResNet, YOLO)
- Language-only: 텍스트 생성, 번역 (GPT, BERT)
- Audio-only: 음성 인식, 음악 생성

### Multimodal AI (다중 모달리티)
여러 데이터 타입을 동시에 이해하고 연결:

- Vision + Language: 이미지 캡셔닝, VQA
- Audio + Language: 음성 어시스턴트
- Vision + Audio + Language: 비디오 이해

--- 

## 2. Multimodal Learning의 핵심 개념

### 2.1 Modality (모달리티)
데이터의 "종류"를 의미

- visual modality : 이미지, 비디오 (2D/3D 픽셀)
- textual modality : 단어, 문장 (discrete tokens)
- audio modality : 음성, 소리 (1D waveform)

각 modality는 다른 구조를 가짐
- 이미지 : (H, W, C) 행렬, 공간적 구조
- 텍스트 : (Seq_len,) 시퀀스, 순차적 구조
- 이질적이라서 직접 비교 불가

### 2.2 Representation (표현)
각 modality를 벡터 공간으로 변환

- Vision representation
    - image → CNN/ViT → [1024-dim vector]

- Language representation  
  - text → BERT/GPT → [768-dim vector]


**문제**: 차원도 다르고, 의미 공간도 다름
- 이미지 벡터의 dim 100 ≠ 텍스트 벡터의 dim 100


### 2.3 Alignment (정렬)
서로 다른 modality를 **같은 의미 공간**으로 매핑:

```
Image: [cat photo] → [v1, v2, ..., v512]
                           ↓ alignment
Text:  "a cat"    → [t1, t2, ..., t512]
```

목표 : 같은 의미면 벡터가 가까워야 함
- cosine_similarity(cat_image, "a cat") > 0.8
- cosine_similarity(cat_image, "a dog") < 0.3

#### Alignment 방법
1. contrastive learning (대조 학습)
- 같은 것끼리는 가깝게, 다른 것끼리는 멀게
2. cross-attention
- 한 Modality가 다른 modality를 참조
- BLIP-2 의 Q-Former가 이 방식
3. simple projection
- Linear layer로 차원만 맞춤
- LLaVA가 이 방식 (가장 단순)

---

## 3. Cross-modal Alignment가 필요한 이유

시나리오 : 이미지 캡셔닝

### Alignment 없이 (naive 방법)
```
# 1. 이미지 분류
image → ResNet → "cat" (label)

# 2. 텍스트 생성
"cat" → GPT → "A cat is sleeping on the couch"
```

문제점
- 실제 이미지엔 소파가 없는데 hallucination 발생
- label만으로는 디테일 전달 안됨 (색깔, 자세, 배경 등)
- 이미지와 텍스트가 진짜 연결 된게 아님

### Alignment 있으면
```
# 1. Image → visual features (spatial information 유지)
image → ViT → [196개 patch features]  # 14x14 grid

# 2. Visual features를 language space로 projection
visual_features → Projector → [196개 language-aligned features]

# 3. LLM이 visual features를 "보면서" 생성
LLM("Describe: " + visual_features) 
    → "An orange tabby cat is sleeping on a gray couch near a window"
```

visual features가 LLM의 embedding space와 aligned되어 있어서
- llm이 이미지의 구체적 내용을 이해
- spatial information 활용 가능
- hallucination 감소

---

## 4. VLM의 동작 원리 (간단한 예시)

### 전체 흐름
```
Input: 
  - Image: [고양이 사진]
  - Text: "이 동물은 무엇인가요?"

Step 1: Vision Encoding
  Image → ViT → Visual Tokens [v1, v2, ..., v196]
  # 각 토큰은 이미지의 한 patch를 표현

Step 2: Projection (Alignment)
  Visual Tokens → Linear Layer → [h1, h2, ..., h196]
  # h_i는 LLM의 embedding space에 있음

Step 3: Token Fusion
  Combined = [BOS] + [h1, ..., h196] + ["이", "동물", "은", ...] + [EOS]
  # Visual tokens와 text tokens를 하나로 합침

Step 4: LLM Decoding
  LLM(Combined) → "이것은 고양이입니다. 주황색과 흰색 털을 가진 
                   태비 고양이로 보입니다."

```

#### 왜 이렇게 작동하나?
핵심 아이디어: LLM은 이미 엄청난 언어 이해 능력이 있음

- GPT, LLaMA 같은 모델은 수조 개의 텍스트로 학습됨
- "고양이", "주황색", "털" 같은 개념을 이미 알고 있음

하지만: 이미지는 본 적이 없음

- 이미지 픽셀을 직접 주면 이해 못함
- "텍스트처럼 보이게" 변환이 필요 → Projection!

---

## 5. Contrastive Learning 상세 설명
CLIP 같은 모델의 핵심 학습 방법으로 기본 아이디어는 "같은 의미는 가깝게, 다른 의미는 멀게"

### 학습 과정
```python
# Batch 데이터
images = [img1, img2, img3, img4]  # 4개 이미지
texts = ["a dog", "a cat", "a car", "a flower"]  # 대응하는 텍스트

# Encoding
image_features = vision_encoder(images)  # [4, 512]
text_features = text_encoder(texts)      # [4, 512]

# Normalize (코사인 유사도 계산을 위해)
image_features = normalize(image_features)
text_features = normalize(text_features)

# Similarity matrix 계산
similarity = image_features @ text_features.T  # [4, 4]
#           text1  text2  text3  text4
# image1  [[0.9,   0.1,   0.05,  0.02],   # img1-text1 매칭
# image2   [0.1,   0.85,  0.03,  0.08],   # img2-text2 매칭
# image3   [0.05,  0.02,  0.9,   0.01],   # img3-text3 매칭
# image4   [0.03,  0.1,   0.02,  0.88]]   # img4-text4 매칭

```

### Loss 계산
```python
# 대각선(올바른 매칭)은 높게, 나머지는 낮게
labels = [0, 1, 2, 3]  # 대각선 인덱스

# Image-to-Text loss
loss_i2t = CrossEntropyLoss(similarity, labels)

# Text-to-Image loss (대칭)
loss_t2i = CrossEntropyLoss(similarity.T, labels)

# Total loss
loss = (loss_i2t + loss_t2i) / 2

```

### 효과
```python
# 테스트
new_image = "고양이 사진"
candidates = ["a cat", "a dog", "a car"]

# Similarity 계산
sims = cosine_similarity(
    vision_encoder(new_image),
    [text_encoder(t) for t in candidates]
)
# 결과: [0.85, 0.2, 0.1]
# → "a cat"이 가장 높음!
```

왜 강력한가?
- 명시적인 label 없이 학습
- zero-shot classification 가능
- 수억 개의 이미지-텍스트 쌍으로 학습 가능 (웹 크롤링)

---

## 6. VLM의 3가지 핵심 컴포넌트
vlm은 구조적으로 보면 3단 연결 구조다.

### 6.1 Vision Encoder
대표 구조:
- CNN (ResNet, ConvNeXt)
- ViT (Vision Transformer)

출력:
- 이미지 전체를 대표하는 embedding
- 또는 patch-level token들

중요 포인트:
- 대부분 사전학습(pretrained)된 모델
- 종종 freeze됨 (특히 LLM 결합 구조에서)

### 6.2 Projection Layer (Bridge)
Vision feature를 Language 공간으로 변환하는 다리

필요한 이유
- Vision encoder 출력 차원 ≠ LLM 입력 차원
- 표현 분포 자체도 다름

역할
- image embedding → LLM이 이해 가능한 token 형태로 변환
- linear / MLP / Q-Former 등 사용

예시
- CLIP: image embedding ↔ text embedding 직접 정렬
- LLaVA/BLIP-2: visual token → language token space로 사상

📌 이 레이어가 cross-modal alignment의 핵심 지점

### 6.3 LLM (Language Model)
최종 이해, 추론, 생성 담당

역할
- 시각 정보 + 텍스트 instruction을 결합
- reasoning 수행
- 자연어 응답 생성

특징
- GPT / LLaMA 계열
- 종종 freeze + projection만 학습
- instruction tuning으로 멀티모달 능력 확보