# Vision-Language Models (VLM) 정리

## 1. VLM 기본 개념

Vision-Language Model(VLM)은 이미지와 텍스트를 **공통 표현 공간에서 이해하고 추론**하는 멀티모달 모델이다.  
이미지를 단순 분류하는 것을 넘어, 텍스트 질의에 대해 이미지를 해석하고 **언어로 응답**할 수 있다.

### 핵심 특징
- Image + Text 동시 처리
- 멀티모달 추론 (reasoning)
- OCR, 문서 이해, 시각적 질의응답(VQA) 가능
- LLM과 결합 시 범용 AI 시스템의 핵심 구성요소

---

## 2. VLM 기본 구조
VLM은 대부분 아래 구조로 수렴한다.

Image → Vision Encoder → Projection Layer → LLM
Text → Tokenizer ------------------------→ LLM

### 구성 요소 설명
- **Vision Encoder**: ViT, ResNet, Swin 등
- **Projection Layer**: visual feature를 language embedding space로 정렬
- **LLM**: 텍스트 생성, 추론 담당


## 3. VLM 모델 타임라인

### 3.1 Alignment 기반 VLM (Contrastive Learning)

| 연도 | 모델 | 특징 |
|----|----|----|
| 2021 | CLIP | Image-Text embedding 정렬 |
| 2021 | ALIGN | 대규모 noisy 데이터 |
| 2022 | BLIP | retrieval + caption 통합 |

- 생성 능력 없음
- zero-shot 분류, 검색에 특화

### 3.2 Encoder–Decoder VLM

| 연도 | 모델 | 특징 |
|----|----|----|
| 2020 | OSCAR | object tag 활용 |
| 2022 | BLIP-2 | Q-Former 도입 |
| 2022 | Flamingo | few-shot VQA |

- Vision → Text 생성 가능
- task-specific 성향 강함


### 3.3 LLM 결합 VLM (Modern VLM)

| 연도 | 모델 | 특징 |
|----|----|----|
| 2023 | LLaVA | CLIP + LLaMA |
| 2023 | MiniGPT-4 | ViT + Vicuna |
| 2023 | InstructBLIP | instruction tuning |
| 2024 | Qwen-VL | OCR + reasoning 강화 |
| 2024 | GPT-4V / GPT-5V | 범용 멀티모달 |

- LLM 중심 구조
- 추론, 대화, OCR, 문서 이해 가능


## 4. 주요 VLM 모델 비교

| 모델 | 생성 | OCR | Reasoning | 오픈소스 |
|----|----|----|----|----|
| CLIP | ❌ | ❌ | ❌ | ⭕ |
| BLIP-2 | ⭕ | ❌ | ⭕ | ⭕ |
| LLaVA | ⭕ | △ | ⭕ | ⭕ |
| InstructBLIP | ⭕ | △ | ⭕ | ⭕ |
| Qwen-VL | ⭕ | ⭕ | ⭕ | ⭕ |
| GPT-4V | ⭕ | ⭕ | ⭕ | ❌ |

---

## 5. 모델별 코드 사용법 (간략)

### 5.1 CLIP
```python
import clip, torch
from PIL import Image

model, preprocess = clip.load("ViT-B/32")
image = preprocess(Image.open("image.jpg")).unsqueeze(0)
text = clip.tokenize(["a dog", "a cat"])

with torch.no_grad():
    logits, _ = model(image, text)
```

### 5.2 BLIP-2
```python
from transformers import Blip2Processor, Blip2ForConditionalGeneration

processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl"
)

inputs = processor(image, "What is in the image?", return_tensors="pt")
outputs = model.generate(**inputs)
```

### 5.3 LLaVA
```python
from llava.model.builder import load_pretrained_model

model, processor = load_pretrained_model("llava-v1.5")
response = model.chat(image, "Describe this image")
```

### 5.4 Qwen-VL
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL")

inputs = tokenizer.from_list_format([
    {"image": "image.jpg"},
    {"text": "이 이미지에 대해 설명해줘"}
])
```

## 주요 개념 요약
- Cross-modal alignment
→ 이미지와 텍스트를 같은 embedding 공간으로 정렬
- Contrastive Learning
→ positive/negative image-text pair 학습
- Q-Former
→ visual 정보를 LLM에 효율적으로 전달하는 중간 모듈
- Instruction Tuning
→ VLM을 대화형 모델로 전환

## 실무에서 사용하는 VLM
문서 이해 / OCR
- GPT-4V / GPT-5V
- Qwen-VL

온프레미스 / PoC
- LLaVA
- BLIP-2

검색 / 분류
- CLIP