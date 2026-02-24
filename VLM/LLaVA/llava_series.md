# LLaVA 시리즈 완벽 정리

LLaVA (Large Language and Vision Assistant) 시리즈의 진화 과정을 정리합니다.

---

## 목차
1. [LLaVA v1.0 (2023.04)](#1-llava-v10-202304)
2. [LLaVA-1.5 (2023.10)](#2-llava-15-202310)
3. [LLaVA-NeXT (2024.01)](#3-llava-next-202401)
4. [LLaVA-OneVision (2024.08)](#4-llava-onevision-202408)
5. [전체 비교](#전체-비교표)
6. [실무 가이드](#실무-선택-가이드)

---

## 1. LLaVA (v1.0, 2023.04)

### 논문 정보
**제목:** "Visual Instruction Tuning"  
**발표:** NeurIPS 2023  
**기관:** University of Wisconsin-Madison, Microsoft Research

### 핵심 아이디어

**Instruction Tuning을 Vision에 적용!**

```
기존 VLM: Image-text pairs로 학습
LLaVA: GPT-4로 생성한 instruction data로 학습
```

### 아키텍처

```
┌─────────────────────────────────────┐
│         Image (224×224)              │
└──────────────┬──────────────────────┘
               ↓
    ┌──────────────────────┐
    │ Vision Encoder        │
    │ CLIP ViT-L/14        │
    │ (frozen)             │
    └──────────┬───────────┘
               ↓
        Visual features
        (256 tokens)
               ↓
    ┌──────────────────────┐
    │ Projection Matrix W   │
    │ (Linear layer)        │
    │ 1024 → 4096          │
    └──────────┬───────────┘
               ↓
        Visual tokens
               ↓
    ┌──────────────────────────────┐
    │ [IMG_START] v1 v2 ... v256   │
    │ [IMG_END] <instruction>      │
    └──────────┬───────────────────┘
               ↓
    ┌──────────────────────┐
    │ LLM (Vicuna-7B/13B)  │
    │ (fine-tuned)         │
    └──────────┬───────────┘
               ↓
          Response
```

### 핵심 특징

#### 1. GPT-4 생성 데이터
```python
# Pseudo-code
coco_image = load_image()
coco_captions = get_captions(coco_image)
coco_objects = get_objects(coco_image)

prompt = f"""
Given this image context:
Captions: {coco_captions}
Objects: {coco_objects}

Generate 3 types of conversations:
1. Detailed description
2. Reasoning questions
3. Complex conversations
"""

instruction_data = gpt4(prompt)
```

**생성 데이터 종류:**
- **Conversation**: 158K conversations (from COCO)
- **Detailed description**: 23K
- **Complex reasoning**: 77K

#### 2. 2-Stage Training

**Stage 1: Pre-training (Feature Alignment)**
```
Goal: Vision feature → Language space 정렬
Data: CC3M (595K image-caption pairs)
Frozen: Vision Encoder + LLM
Trainable: Projection matrix W only

Loss: Next token prediction on captions
```

**Stage 2: Fine-tuning (Instruction Tuning)**
```
Goal: Instruction following 능력
Data: 158K GPT-4 generated instruction data
Frozen: Vision Encoder
Trainable: Projection + LLM

Loss: Next token prediction on responses
```

### 성능

| Benchmark | LLaVA-13B | GPT-4V |
|:----------|:----------|:-------|
| VQAv2 | 80.0 | - |
| GQA | 62.0 | - |
| ScienceQA | 90.9 | - |

---

## 2. LLaVA-1.5 (2023.10)

### 논문 정보
**제목:** "Improved Baselines with Visual Instruction Tuning"  
**발표:** Technical Report  
**개선:** 간단한 수정으로 큰 성능 향상

### 주요 개선 사항

#### 1. MLP Projector (핵심!)
```python
# LLaVA v1.0: Linear projection
projection = nn.Linear(1024, 4096)

# LLaVA-1.5: 2-layer MLP
projection = nn.Sequential(
    nn.Linear(1024, 4096),
    nn.GELU(),
    nn.Linear(4096, 4096)
)
```

**왜 개선?**
- Linear는 너무 단순
- MLP로 비선형 변환 → 더 나은 alignment
- 성능 10% 향상!

#### 2. 고해상도 이미지 (336×336)
```
LLaVA v1.0: 224×224 (CLIP 기본)
LLaVA-1.5: 336×336

Patches: 224/14 = 16×16 = 256 tokens
      → 336/14 = 24×24 = 576 tokens
```

**효과:**
- 작은 객체/텍스트 인식 개선
- OCR 능력 향상
- Detail understanding 개선

#### 3. 더 나은 학습 데이터

**Academic Task-oriented VQA Mix:**
```
LLaVA v1.0: Only conversation data
LLaVA-1.5: 
- Conversation (665K)
- VQAv2 (83K)
- GQA (72K)
- OKVQA (9K)
- OCRVQA (80K)
- A-OKVQA (17K)
→ Total: 665K
```

**효과:**
- Benchmark 성능 대폭 향상
- Zero-shot capability 개선

#### 4. 더 강력한 LLM

```
LLaVA v1.0: Vicuna-7B/13B
LLaVA-1.5: Vicuna-7B/13B + Mistral-7B 지원
```

### 아키텍처 비교

```
LLaVA v1.0:
Image (224²) → CLIP → Linear → LLM (Vicuna)

LLaVA-1.5:
Image (336²) → CLIP → MLP (2-layer) → LLM (Vicuna/Mistral)
              ↑                ↑
          더 고해상도      비선형 변환
```

### 성능 향상

| Benchmark | LLaVA v1.0 | LLaVA-1.5 | 개선 |
|:----------|:-----------|:----------|:-----|
| **VQAv2** | 80.0 | **85.1** | +5.1 |
| **GQA** | 62.0 | **67.4** | +5.4 |
| **POPE (Hallucination)** | 85.9 | **86.4** | +0.5 |
| **MMBench** | 64.3 | **68.2** | +3.9 |
| **MM-Vet** | 30.5 | **35.4** | +4.9 |

**핵심: 간단한 변경으로 큰 성능 향상!**

---

## 3. LLaVA-NeXT (2024.01)

### 논문 정보
**제목:** "LLaVA-NeXT: Improved reasoning, OCR, and world knowledge"  
**발표:** Technical Report  
**별칭:** LLaVA-1.6

### 주요 혁신

#### 1. Dynamic High-Resolution (가장 중요!)

**문제:**
```
고정 해상도 (336×336):
- 긴 이미지 (문서, 포스터) → squash됨
- 넓은 이미지 (파노라마) → squash됨
→ Detail 손실
```

**해결: AnyRes (Any Resolution)**

```python
# Pseudo-code
def process_image_anyres(image, base_size=336, max_tiles=5):
    """
    이미지를 base_size의 grid로 분할
    """
    aspect_ratio = image.width / image.height
    
    # 최적 grid 찾기
    if aspect_ratio > 1.5:  # Wide image
        grid = (3, 1)  # 3개 가로 타일
    elif aspect_ratio < 0.67:  # Tall image
        grid = (1, 3)  # 3개 세로 타일
    else:
        grid = (2, 2)  # 2×2 grid
    
    # 각 타일 처리
    tiles = split_image_to_tiles(image, grid, base_size)
    
    tile_features = []
    for tile in tiles:
        feat = vision_encoder(tile)  # 각 타일 독립 encoding
        tile_features.append(feat)
    
    # Global view도 추가
    global_tile = resize(image, base_size)
    global_feat = vision_encoder(global_tile)
    
    # Concatenate: [global, tile1, tile2, ...]
    all_features = [global_feat] + tile_features
    
    return all_features
```

**구조:**
```
Original Image (1920×1080, wide)
    ↓
Split into 3×1 grid + 1 global
    ↓
┌─────┬─────┬─────┐  ┌─────┐
│Tile1│Tile2│Tile3│  │Global│
│336² │336² │336² │  │336²  │
└─────┴─────┴─────┘  └─────┘
   ↓     ↓     ↓        ↓
Vision Encoder (CLIP) 각각 처리
   ↓     ↓     ↓        ↓
 576   576   576      576 tokens
   └─────┴─────┴────────┘
            ↓
    [Global + Tile1 + Tile2 + Tile3]
            ↓
      Total: 576×4 = 2304 tokens
            ↓
          LLM
```

**효과:**
- **OCR 대폭 향상** (작은 텍스트)
- **문서 이해** (긴 문서)
- **Detail preservation**

#### 2. 더 강력한 LLM 지원

```
LLaVA-1.5: Vicuna-7B/13B, Mistral-7B
LLaVA-NeXT: 
- Vicuna-7B/13B
- Mistral-7B
- Yi-34B          ← 새로 추가
- Nous-Hermes-34B ← 새로 추가
- Mixtral-8×7B (56B) ← MoE!
```

**Mixtral 사용의 장점:**
- 56B 파라미터 (실제 사용은 13B)
- MoE로 효율적
- 더 강력한 reasoning

#### 3. 개선된 학습 데이터

**1M+ diverse data mix:**
```
Conversation: 
- LLaVA-Instruct-665K
- ShareGPT-4V (100K)

Academic VQA:
- All previous tasks
+ DocVQA (39K) ← Document 특화
+ ChartQA (18K) ← Chart 이해
+ SynDog (50K) ← OCR synthetic

Total: ~1M samples
```

#### 4. Better Training Recipe

**3-Stage Training:**

**Stage 1: Feature Alignment (same as before)**
```
Data: 558K (LAION, CC, SBU)
Trainable: Projector only
Duration: 1 epoch
```

**Stage 2: High-quality Alignment**
```
Data: LCS-558K (filtered, high-quality subset)
Trainable: Projector + LLM
Duration: 1 epoch
```

**Stage 3: Instruction Fine-tuning**
```
Data: 1M+ mixed task data
Trainable: Projector + LLM
Duration: 1 epoch
```

### 성능

| Benchmark | LLaVA-1.5 | LLaVA-NeXT-34B | GPT-4V |
|:----------|:----------|:---------------|:-------|
| **MMBench** | 68.2 | **79.3** | 77.0 |
| **POPE** | 86.4 | **87.2** | - |
| **MM-Vet** | 35.4 | **51.1** | 56.8 |
| **TextVQA** | 61.3 | **69.5** | 78.0 |
| **DocVQA** | - | **69.5** | 88.4 |
| **ChartQA** | - | **65.8** | 78.5 |

**핵심: Dynamic resolution으로 OCR과 document 이해 대폭 향상!**

---

## 4. LLaVA-OneVision (2024.08)

### 논문 정보
**제목:** "LLaVA-OneVision: Easy Visual Task Transfer"  
**발표:** Technical Report  
**핵심:** Single model for Image + Video + 3D

### 주요 혁신

#### 1. Unified Vision Encoder

**문제:**
```
기존: CLIP (Image only)
→ Video, 3D는?
```

**해결: SigLIP (Google)**
```
CLIP의 개선 버전:
- Better image-text alignment
- More robust
- Video frame도 잘 처리

LLaVA-OneVision: SigLIP-SO400M
- 400M parameter vision encoder
- 384×384 resolution
```

#### 2. Unified Task Training

**Single model for 4 modalities:**

```
┌─────────────────────────────────────────┐
│         LLaVA-OneVision                  │
├─────────────────────────────────────────┤
│                                          │
│  Images ──┐                             │
│  Video ───┤→ SigLIP → Projector → LLM  │
│  3D ──────┘                             │
│                                          │
└─────────────────────────────────────────┘

Output:
- Image understanding
- Video understanding (temporal)
- 3D scene understanding
```

**Video 처리:**
```python
def process_video(video, num_frames=8):
    """
    Video → uniform sampling → frames
    """
    frames = uniformly_sample(video, num_frames)
    
    frame_features = []
    for frame in frames:
        feat = vision_encoder(frame)  # Each frame
        frame_features.append(feat)
    
    # Temporal encoding
    # Option 1: Concatenate all frames
    # Option 2: Pooling
    # Option 3: Learnable temporal tokens
    
    video_tokens = aggregate(frame_features)
    return video_tokens
```

**3D 처리:**
```python
def process_3d_scene(point_cloud):
    """
    3D scene → multiple views → images
    """
    views = render_multiple_views(point_cloud, num_views=6)
    # Front, back, left, right, top, bottom
    
    view_features = []
    for view in views:
        feat = vision_encoder(view)
        view_features.append(feat)
    
    scene_tokens = aggregate(view_features)
    return scene_tokens
```

#### 3. Task Transfer Learning

**Single-image tasks:**
- VQA, captioning, OCR, ...
- 기존 LLaVA-NeXT 능력 유지

**Multi-image tasks (new!):**
```
Input: [IMG1] [IMG2] ... Compare these images
Output: Comparison, differences, relationships
```

**Video tasks (new!):**
```
Input: [Video frames] What happens in this video?
Output: Action recognition, temporal reasoning
```

**3D tasks (new!):**
```
Input: [3D scene from multiple views] Describe this room
Output: 3D spatial understanding
```

#### 4. Training Strategy

**Stage 0: Vision Encoder Adaptation (optional)**
```
SigLIP을 video/3D data로 fine-tune
→ Better feature quality
```

**Stage 1-3: Same as LLaVA-NeXT**
```
But with mixed data:
- Images (70%)
- Video (20%)
- 3D (10%)
```

**Key insight: Task transfer**
```
Image understanding 능력이 Video/3D로 transfer됨!
→ Video/3D data 적게 필요
```

#### 5. OneVision-specific Features

**Temporal Reasoning:**
```
Video: [Frame1, Frame2, ..., Frame8]
Question: "What happens after the person picks up the cup?"
→ Temporal order understanding
```

**Spatial Reasoning (3D):**
```
3D scene: [View1, View2, ..., View6]
Question: "What's on the left of the sofa?"
→ 3D spatial relationships
```

**Cross-modal Transfer:**
```
Strong image understanding
  ↓ (transfer)
Video frame understanding
  ↓ (aggregate)
Temporal understanding
```

### 아키텍처 비교

```
LLaVA-NeXT:
Image (dynamic res) → CLIP → MLP → LLM

LLaVA-OneVision:
┌── Image (dynamic res) ──┐
├── Video (frames) ────────┤→ SigLIP → MLP → LLM
└── 3D (multi-view) ──────┘
```

### 성능

**Image tasks (유지/개선):**
| Benchmark | LLaVA-NeXT | OneVision |
|:----------|:-----------|:----------|
| MMBench | 79.3 | **80.1** |
| MM-Vet | 51.1 | **52.3** |

**Video tasks (new capability!):**
| Benchmark | Performance |
|:----------|:------------|
| **Video-MME** | **66.2** |
| **ActivityNet-QA** | **56.7** |
| **NeXT-QA** | **71.3** |

**Multi-image tasks:**
| Benchmark | Performance |
|:----------|:------------|
| **NLVR2** | **86.9** |
| **Q-Bench** | **79.4** |

---

## 전체 비교표

| Feature | LLaVA v1.0 | LLaVA-1.5 | LLaVA-NeXT | OneVision |
|:--------|:-----------|:----------|:-----------|:----------|
| **발표** | 2023.04 | 2023.10 | 2024.01 | 2024.08 |
| **Vision Encoder** | CLIP ViT-L/14 | CLIP ViT-L/14 | CLIP ViT-L/14 | SigLIP-SO400M |
| **해상도** | 224×224 | 336×336 | Dynamic (AnyRes) | 384×384 + Dynamic |
| **Projector** | Linear | 2-layer MLP | 2-layer MLP | 2-layer MLP |
| **LLM** | Vicuna 7B/13B | Vicuna + Mistral | + Yi-34B, Mixtral | Qwen-2 (0.5B-72B) |
| **학습 데이터** | 158K GPT-4 | 665K mixed | 1M+ mixed | 1M+ multi-modal |
| **지원 모달리티** | Image only | Image only | Image only | Image + Video + 3D |
| **특화 능력** | Instruction | Better VQA | OCR, Document | Temporal, Spatial |
| **MMBench** | 64.3 | 68.2 | 79.3 | 80.1 |
| **MM-Vet** | 30.5 | 35.4 | 51.1 | 52.3 |
| **TextVQA** | - | 61.3 | 69.5 | 70.2 |

---

## 주요 혁신 정리

### LLaVA v1.0 (2023.04)
```
혁신: Visual Instruction Tuning
- GPT-4로 데이터 생성
- 2-stage training
→ VLM의 새로운 패러다임
```

### LLaVA-1.5 (2023.10)
```
혁신: Simple but Effective
- Linear → MLP projector
- 224 → 336 resolution
- Better data mix
→ 10% 성능 향상, minimal changes
```

### LLaVA-NeXT (2024.01)
```
혁신: Dynamic High-Resolution
- AnyRes: 이미지를 tile로 분할
- 34B LLM 지원
- Document/OCR 특화
→ Detail understanding 대폭 향상
```

### LLaVA-OneVision (2024.08)
```
혁신: Unified Multi-modal
- Image + Video + 3D
- SigLIP encoder
- Task transfer learning
→ Single model for all vision tasks
```

---

## 진화 트렌드

### 1. 해상도 진화
```
224×224 (v1.0)
  ↓
336×336 (v1.5)
  ↓
Dynamic AnyRes (NeXT)
  ↓
384×384 + Dynamic (OneVision)
```

### 2. 모델 크기 확장
```
7B/13B (v1.0)
  ↓
+ Mistral-7B (v1.5)
  ↓
+ Yi-34B, Mixtral-56B (NeXT)
  ↓
Qwen-2 0.5B ~ 72B (OneVision)
```

### 3. 모달리티 확장
```
Image (v1.0)
  ↓
Image (v1.5)
  ↓
Image + better detail (NeXT)
  ↓
Image + Video + 3D (OneVision)
```

### 4. 데이터 확장
```
158K GPT-4 (v1.0)
  ↓
665K mixed (v1.5)
  ↓
1M+ diverse (NeXT)
  ↓
1M+ multi-modal (OneVision)
```

---

## 실무 선택 가이드

### 일반 VQA / Conversation
```
→ LLaVA-1.5-7B
이유: 빠르고, 충분한 성능, 경량
```

### OCR / Document Understanding
```
→ LLaVA-NeXT-34B
이유: Dynamic resolution, document 특화
```

### Video Understanding
```
→ LLaVA-OneVision
이유: Temporal reasoning, video 특화
```

### 최고 성능 필요
```
→ LLaVA-NeXT-34B or OneVision-72B
이유: 대형 LLM, 최고 정확도
```

### 리소스 제한
```
→ LLaVA-1.5-7B or OneVision-0.5B
이유: 경량, on-device 가능
```

---

## 코드 예시

### LLaVA-1.5 사용
```python
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images
from PIL import Image

model_path = "liuhaotian/llava-v1.5-7b"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

image = Image.open("image.jpg")
image_tensor = process_images([image], image_processor, model.config)

prompt = "Describe this image in detail."
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX)

with torch.no_grad():
    output_ids = model.generate(
        input_ids.unsqueeze(0).cuda(),
        images=image_tensor.cuda(),
        max_new_tokens=512,
        use_cache=True
    )

output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
print(output)
```

### LLaVA-NeXT Dynamic Resolution
```python
# AnyRes processing
def process_anyres_image(image, processor, grid_size=(2,2)):
    """
    Split image into tiles and process
    """
    tiles = []
    
    # Global view
    global_image = image.resize((336, 336))
    tiles.append(processor(global_image))
    
    # Split into grid
    width, height = image.size
    tile_width = width // grid_size[0]
    tile_height = height // grid_size[1]
    
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            tile = image.crop((
                i * tile_width,
                j * tile_height,
                (i+1) * tile_width,
                (j+1) * tile_height
            ))
            tile = tile.resize((336, 336))
            tiles.append(processor(tile))
    
    return torch.stack(tiles)
```

### LLaVA-OneVision Video
```python
def process_video(video_path, model, num_frames=8):
    """
    Process video for LLaVA-OneVision
    """
    # Load and sample frames
    frames = sample_frames_uniformly(video_path, num_frames)
    
    # Process each frame
    frame_tensors = []
    for frame in frames:
        tensor = image_processor(frame)
        frame_tensors.append(tensor)
    
    video_tensor = torch.stack(frame_tensors)
    
    # Generate
    prompt = "<video>\nWhat happens in this video?"
    output = model.generate(prompt, video=video_tensor)
    
    return output
```

---

## 요약

**LLaVA 시리즈는 오픈소스 VLM의 표준이 되었으며, 간단한 구조와 강력한 성능으로 연구와 실무 모두에서 널리 사용되고 있습니다.**

### 핵심 포인트
- **v1.0**: Visual Instruction Tuning의 시작
- **v1.5**: 간단한 개선으로 10% 성능 향상
- **NeXT**: Dynamic resolution으로 OCR/Document 특화
- **OneVision**: Image + Video + 3D 통합

### 미래 방향
- 더 큰 해상도와 더 긴 context
- 더 효율적인 architecture
- 더 다양한 modality 통합
- End-to-end 3D understanding