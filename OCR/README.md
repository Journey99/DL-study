# OCR (Optical Character Recognition) 완벽 정리

## 목차
1. [OCR 개념 및 구조](#1-ocr-개념-및-구조)
2. [OCR 발전 역사](#2-ocr-발전-역사)
3. [주요 모델 변천사](#3-주요-모델-변천사)
4. [현재 많이 사용하는 모델](#4-현재-많이-사용하는-모델)
5. [도메인별 모델 선택](#5-도메인별-모델-선택)
6. [실무 가이드](#6-실무-가이드)

---

## 1. OCR 개념 및 구조

### 1.1 OCR이란?
**Optical Character Recognition (광학 문자 인식)**
- 이미지 속 텍스트를 기계가 읽을 수 있는 문자로 변환
- 스캔 문서, 사진, 스크린샷 등에서 텍스트 추출

### 1.2 OCR 파이프라인

```
Input Image
    ↓
Text Detection (어디에 텍스트가 있나?)
    ↓
Text Recognition (무슨 글자인가?)
    ↓
Post-processing (언어 모델, 보정)
    ↓
Output Text
```

### 1.3 OCR의 두 가지 주요 Task

| Task | 목적 | 출력 | 대표 모델 |
|:-----|:-----|:-----|:----------|
| **Text Detection** | 텍스트 영역 찾기 | Bounding box/Polygon | EAST, CRAFT, DBNet |
| **Text Recognition** | 텍스트 읽기 | 문자열 | CRNN, ASTER, ABINet |

### 1.4 OCR 분류

#### 응용 분야별
- **Scene Text OCR**: 자연 이미지 속 텍스트 (간판, 표지판)
- **Document OCR**: 스캔 문서, PDF (책, 논문, 계약서)
- **Handwriting Recognition**: 필기체 인식

#### 처리 방식별
- **Two-Stage**: Detection → Recognition (전통적)
- **End-to-End**: 한 번에 처리 (최신 트렌드)
- **OCR-free**: Recognition 없이 직접 이해 (Donut)

---

## 2. OCR 발전 역사

### 2.1 시대별 발전 과정

#### 1세대: 규칙 기반 (1950s-1990s)
- **특징**: Template matching, 패턴 인식
- **대표**: 초기 Tesseract (v1-v3)
- **한계**: 폰트, 크기 변화에 취약

#### 2세대: 전통 Machine Learning (2000s-2014)
- **특징**: Feature extraction + SVM/HMM
- **대표**: Tesseract v3 (2007)
- **개선**: 다양한 폰트 처리 가능
- **한계**: Feature engineering 필요

#### 3세대: Deep Learning 초기 (2014-2017)
- **특징**: CNN + RNN 조합
- **대표**: 
  - **CRNN** (2015): CNN + LSTM + CTC
  - **EAST** (2017): FCN 기반 detection
- **혁신**: End-to-end 학습, Feature 자동 학습
- **한계**: Curved text, irregular layout 약함

#### 4세대: Attention 시대 (2017-2020)
- **특징**: Attention mechanism 도입
- **대표**:
  - **CRAFT** (2019): Character-level attention
  - **ASTER** (2018): STN + Attention
  - **SAR** (2019): 2D Attention
- **혁신**: 복잡한 layout, distortion 처리
- **한계**: 긴 텍스트, context 이해 부족

#### 5세대: Transformer 시대 (2020-현재)
- **특징**: Pure Transformer 또는 Hybrid
- **대표**:
  - **TrOCR** (2021): ViT + Text Transformer
  - **ABINet** (2021): Vision + Language model
  - **PARSeq** (2022): Permutation LM
  - **Donut** (2022): OCR-free document understanding
- **혁신**: Context 이해, Pre-training 활용
- **현재**: SOTA 성능, 다양한 도메인 적용

### 2.2 패러다임 전환

| 시기 | 패러다임 | 핵심 기술 | 예시 |
|:-----|:---------|:---------|:-----|
| ~2014 | Rule-based | Template matching | Tesseract v3 |
| 2015-2017 | CNN+RNN | CTC Loss | CRNN |
| 2018-2020 | Attention | Seq2Seq | ASTER, CRAFT |
| 2021~ | Transformer | Pre-training | TrOCR, Donut |

---

## 3. 주요 모델 변천사

### 3.1 Text Detection 모델 발전

#### EAST (2017) - 실시간 Detection의 시작
- **구조**: FCN 기반
- **특징**: 
  - Single-stage, fast
  - Rotated box 지원
  - Real-time 가능 (13.2ms)
- **의의**: Scene text detection의 표준

#### PixelLink (2018) - Instance Segmentation 방식
- **구조**: Pixel-level segmentation
- **특징**: 
  - Arbitrary shape text
  - Link prediction (픽셀 연결)
- **의의**: Segmentation 기반 접근

#### CRAFT (2019) - Character-level Detection ⭐
- **구조**: U-Net 기반
- **특징**:
  - Character region heatmap
  - Affinity map (글자 간 연결)
  - Weakly-supervised learning
  - Curved text 처리 가능
- **의의**: 현재까지 가장 널리 사용
- **성능**: ICDAR15 90.0 F1-score

#### DBNet/DBNet++ (2020/2022) - Differentiable Binarization
- **구조**: ResNet + FPN
- **특징**:
  - Differentiable binarization
  - Adaptive threshold
  - Real-time + High accuracy
- **의의**: PaddleOCR의 기본 detector
- **성능**: ICDAR15 91.5 F1-score

### 3.2 Text Recognition 모델 발전

#### CRNN (2015) - Recognition의 기초 ⭐
```
CNN (Feature extraction)
  ↓
RNN/LSTM (Sequence modeling)
  ↓
CTC (Decoding)
```
- **의의**: OCR recognition의 표준 구조
- **장점**: End-to-end, 가변 길이 처리
- **한계**: Context 정보 부족

#### Attention-based (2016-2018) - Seq2Seq 도입
- **Show, Attend and Read** (2016)
- **구조**: Encoder-Decoder + Attention
- **특징**: CTC 없이 직접 문자 예측
- **장점**: Alignment 자동 학습

#### ASTER (2018) - Rectification 도입
```
STN (Spatial Transformer Network)
  ↓ (이미지 rectify)
Recognition Network (Attention-based)
```
- **혁신**: Curved/distorted text 처리
- **성능**: Irregular text에서 우수

#### SAR (2019) - 2D Attention
- **구조**: 2D attention mechanism
- **특징**: 
  - Spatial 정보 더 잘 활용
  - Irregular text 강함

#### ABINet (2021) - Language Model 통합 ⭐
```
Vision Model (BiLSTM)
  ↓
Language Model (Transformer)
  ↓
Fusion (Iterative correction)
```
- **혁신**: Vision + Language 통합
- **특징**:
  - Occluded text 복원
  - Context 이해
  - Iterative refinement
- **성능**: 여러 벤치마크 SOTA

#### PARSeq (2022) - Permutation Language Model
- **구조**: Transformer encoder-decoder
- **특징**:
  - Permutation language modeling
  - Context-aware prediction
  - Single model for multiple tasks
- **성능**: 7개 벤치마크에서 SOTA

#### SVTR (2022) - Simple ViT
- **구조**: Vision Transformer 기반
- **특징**:
  - 단순하고 효과적
  - Multi-scale features
- **성능**: 빠르면서 정확

### 3.3 End-to-End 모델 발전

#### Mask TextSpotter (2018/v3 2020)
- **구조**: Mask R-CNN 기반
- **특징**: Detection + Recognition 동시
- **방식**: Instance segmentation

#### ABCNet (2020) / ABCNetv2 (2021)
- **구조**: Bezier curve representation
- **특징**:
  - Arbitrary shape text
  - Real-time capable
  - End-to-end differentiable

#### TESTR (2022) - Transformer Spotting
- **구조**: DETR 방식
- **특징**:
  - Query-based
  - Transformer end-to-end
  - No NMS

### 3.4 Document OCR 전문 모델

#### Tesseract (1985-현재)
- **발전**:
  - v3 (2007): Traditional ML
  - v4 (2018): LSTM 도입
  - v5 (2021): 다국어 개선
- **특징**: 
  - 오픈소스 표준
  - 100+ 언어 지원
  - 문서 OCR에 최적화

#### LayoutLM 시리즈 (2020-2022, Microsoft)
- **LayoutLM** (2020): BERT + Layout
- **LayoutLMv2** (2021): Visual features 추가
- **LayoutLMv3** (2022): Unified architecture
- **특징**:
  - Document understanding
  - Layout 정보 활용
  - Form, Invoice, Receipt 처리

#### TrOCR (2021, Microsoft) - Transformer OCR ⭐
```
Vision Encoder (ViT/DeiT)
  ↓
Text Decoder (RoBERTa)
```
- **특징**:
  - Pre-trained ViT + Language model
  - End-to-end Transformer
  - Handwriting 우수
- **장점**: Fine-tuning 쉬움, HuggingFace 지원

#### Donut (2022) - OCR-free ⭐
- **혁신**: OCR 없이 직접 문서 이해
- **구조**: Swin Transformer + BART
- **특징**:
  - Document classification
  - Information extraction
  - VQA (Visual Question Answering)
- **장점**: OCR 에러 전파 없음

#### Nougat (2023, Meta)
- **목적**: Scientific PDF → Markdown
- **특징**:
  - LaTeX 수식 처리
  - Table, Figure 이해
  - 학술 논문 특화

---

## 4. 현재 많이 사용하는 모델

### 4.1 실무 사용 빈도 Top 5

#### 🥇 1위: EasyOCR
```python
import easyocr
reader = easyocr.Reader(['ko', 'en'])
results = reader.readtext('image.jpg')
```

**사용률:** ⭐⭐⭐⭐⭐

**장점:**
- 설치 및 사용 매우 간단
- 80+ 언어 지원
- Detection (CRAFT) + Recognition 통합
- GPU 가속 지원
- 활발한 커뮤니티

**단점:**
- 커스터마이징 제한적
- 속도 최적화 여지

**사용 사례:**
- 프로토타입 개발
- 다국어 OCR
- Scene text reading

---

#### 🥈 2위: PaddleOCR
```python
from paddleocr import PaddleOCR
ocr = PaddleOCR(lang='korean')
result = ocr.ocr('image.jpg')
```

**사용률:** ⭐⭐⭐⭐⭐

**장점:**
- 매우 빠른 속도 (최적화 우수)
- 높은 정확도
- PP-OCR, PP-OCRv2, PP-OCRv3 시리즈
- 모바일 배포 지원
- 산업계 표준

**단점:**
- 중국어 문서가 많음
- PaddlePaddle 프레임워크 의존

**사용 사례:**
- 프로덕션 배포
- 모바일 앱
- 실시간 처리

---

#### 🥉 3위: Tesseract
```python
import pytesseract
from PIL import Image

text = pytesseract.image_to_string(Image.open('image.jpg'), lang='kor+eng')
```

**사용률:** ⭐⭐⭐⭐

**장점:**
- 오픈소스 원조
- 100+ 언어 지원
- 문서 OCR에 강함
- PDF 처리 가능
- 커스터마이징 가능 (학습 가능)

**단점:**
- Scene text 약함
- 전처리 필수
- 속도 느림

**사용 사례:**
- 문서 디지타이제이션
- PDF 텍스트 추출
- 책/논문 스캔

---

#### 4위: TrOCR (Hugging Face)
```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')

pixel_values = processor(images=image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

**사용률:** ⭐⭐⭐⭐

**장점:**
- Transformer 기반 SOTA
- Handwriting 우수
- Fine-tuning 쉬움
- Pre-trained 모델 풍부
- HuggingFace 생태계

**단점:**
- Recognition only (Detection 별도)
- GPU 필수
- 추론 속도 느림

**사용 사례:**
- 필기체 인식
- 고품질 document OCR
- 연구 및 실험

---

#### 5위: MMOCR
```python
from mmocr.apis import MMOCRInferencer

ocr = MMOCRInferencer(det='DBNet', rec='ABINet')
result = ocr('image.jpg')
```

**사용률:** ⭐⭐⭐

**장점:**
- 거의 모든 OCR 모델 구현
- Config 기반 실험
- 연구용 최적
- OpenMMLab 생태계

**단점:**
- 설정 복잡
- 실무 배포 어려움
- 학습 곡선 높음

**사용 사례:**
- OCR 연구
- 모델 비교 실험
- SOTA 모델 테스트

---

### 4.2 도메인별 최고 모델

| 도메인 | 추천 모델 | 이유 |
|:-------|:---------|:-----|
| **Scene Text** | EasyOCR, PaddleOCR | 범용성, 정확도 |
| **Document** | Tesseract, TrOCR | 문서 특화 |
| **Handwriting** | TrOCR, IAM-based models | Transformer 우수 |
| **Form/Invoice** | LayoutLMv3, Donut | Layout 이해 |
| **Scientific PDF** | Nougat | LaTeX, 수식 처리 |
| **Real-time** | PaddleOCR-mobile | 속도 최적화 |
| **다국어** | EasyOCR | 80+ 언어 |

### 4.3 언어별 추천

#### 한국어
1. **EasyOCR** - 범용성
2. **PaddleOCR** - 속도
3. **Naver Clova OCR** - 상용 (가장 정확)
4. **Pororo** - 한국어 NLP 통합

#### 영어
1. **TrOCR** - SOTA
2. **Tesseract** - 문서
3. **PaddleOCR** - 실시간

#### 중국어
1. **PaddleOCR** - 중국 Baidu
2. **ChineseOCR** - 특화
3. **MMOCR** - 연구

#### 일본어
1. **manga-ocr** - 만화
2. **PaddleOCR** - 범용
3. **EasyOCR** - 간단

---

## 5. 도메인별 모델 선택

### 5.1 Scene Text OCR

**특징:**
- 자연 이미지 속 텍스트
- 다양한 폰트, 크기, 각도
- 조명, 그림자, 왜곡

**최적 조합:**
```
Detection: CRAFT or DBNet
Recognition: ABINet or PARSeq
Framework: EasyOCR or PaddleOCR
```

**사용 사례:**
- 간판, 표지판 인식
- AR 번역
- 자율주행 (교통표지)

### 5.2 Document OCR

**특징:**
- 스캔 문서, PDF
- 정형화된 레이아웃
- 높은 해상도

**최적 모델:**
```
일반 문서: Tesseract, TrOCR
Form/Invoice: LayoutLMv3, Donut
학술 논문: Nougat
```

**사용 사례:**
- 계약서 디지타이제이션
- 청구서 자동화
- 논문 텍스트 추출

### 5.3 Handwriting Recognition

**특징:**
- 필기체 다양성
- 낮은 품질
- Context 중요

**최적 모델:**
```
인쇄된 손글씨: TrOCR (trocr-base-handwritten)
자유로운 필기: IAM-based CRNN
역사적 문서: Transkribus
```

**사용 사례:**
- 설문조사 디지타이제이션
- 역사 문서 보존
- 손편지 디지털화

### 5.4 Real-time OCR

**특징:**
- 모바일/엣지 디바이스
- 제한된 리소스
- 빠른 응답 필요

**최적 모델:**
```
Mobile: PaddleOCR-mobile (PP-OCRv3)
Edge: DBNet-MobileNetV3 + CRNN-tiny
Web: EasyOCR (lightweight mode)
```

**사용 사례:**
- 모바일 스캐너 앱
- 실시간 번역
- POS 시스템

---

## 6. 실무 가이드

### 6.1 빠른 시작 가이드

#### 단순 텍스트 추출
```python
# 가장 간단 - EasyOCR
import easyocr
reader = easyocr.Reader(['ko', 'en'])
result = reader.readtext('image.jpg')
print(result)
```

#### 고정확도 필요
```python
# PaddleOCR
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='korean')
result = ocr.ocr('image.jpg', cls=True)
```

#### 문서 처리
```python
# Tesseract
import pytesseract
from PIL import Image
text = pytesseract.image_to_string(Image.open('doc.jpg'), lang='kor+eng')
```

#### 필기체 인식
```python
# TrOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
```

### 6.2 성능 최적화 팁

#### 전처리
```python
import cv2

# 1. Grayscale 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. Noise 제거
denoised = cv2.fastNlMeansDenoising(gray)

# 3. Binarization
_, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 4. Deskew (기울기 보정)
# ... skew detection and correction
```

#### 후처리
```python
# 1. 언어 모델 적용
from symspellpy import SymSpell
sym_spell = SymSpell()
corrected = sym_spell.lookup(text, Verbosity.CLOSEST)

# 2. 정규 표현식 정제
import re
text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)

# 3. Confidence filtering
results = [r for r in results if r['confidence'] > 0.5]
```

### 6.3 모델 선택 Decision Tree

```
목적이 무엇인가?
├─ Scene Text (자연 이미지)
│  ├─ 빠른 프로토타입 → EasyOCR
│  ├─ 프로덕션 배포 → PaddleOCR
│  └─ 최고 정확도 → CRAFT + ABINet (MMOCR)
│
├─ Document (문서)
│  ├─ 일반 문서 → Tesseract or TrOCR
│  ├─ Form/Invoice → LayoutLMv3 or Donut
│  └─ 학술 논문 → Nougat
│
├─ Handwriting (필기체)
│  ├─ 인쇄된 손글씨 → TrOCR (printed)
│  └─ 자유 필기 → TrOCR (handwritten)
│
└─ Real-time (실시간)
   ├─ Mobile → PaddleOCR-mobile
   └─ Web → EasyOCR (lightweight)
```

### 6.4 평가 지표

| 지표 | 설명 | 사용처 |
|:-----|:-----|:-------|
| **Precision** | 예측한 것 중 맞춘 비율 | Detection |
| **Recall** | 실제 중 찾은 비율 | Detection |
| **F1-score** | Precision과 Recall 조화평균 | Detection |
| **CER** | Character Error Rate | Recognition |
| **WER** | Word Error Rate | Recognition |
| **Accuracy** | 정확히 맞춘 비율 | Recognition |
| **1-NED** | Normalized Edit Distance | Recognition |

### 6.5 Common Issues & Solutions

| 문제 | 원인 | 해결책 |
|:-----|:-----|:-------|
| 낮은 정확도 | 이미지 품질 | 전처리 강화, 해상도 향상 |
| 느린 속도 | 모델 크기 | 경량 모델, 배치 처리 |
| 특수 문자 오류 | 학습 데이터 부족 | Fine-tuning, 후처리 |
| Layout 오류 | Detection 실패 | Detection 모델 개선 |
| 다국어 혼재 | 단일 언어 모델 | 다국어 모델, 언어 감지 |

### 6.6 배포 고려사항

#### On-premise
- **모델**: PaddleOCR, Tesseract
- **장점**: 데이터 보안
- **단점**: 인프라 관리

#### Cloud API
- **서비스**: Google Vision API, AWS Textract, Naver Clova
- **장점**: 관리 불필요, 높은 정확도
- **단점**: 비용, 인터넷 의존

#### Edge/Mobile
- **모델**: PaddleOCR-mobile, CRNN-tiny
- **장점**: 오프라인 가능, 빠름
- **단점**: 정확도 trade-off

---

## 7. 참고 자료

### 주요 벤치마크 데이터셋
- **ICDAR**: Scene text (2013, 2015, 2017, 2019)
- **COCO-Text**: Natural images
- **SVT**: Street View Text
- **IIIT5K**: Scene text
- **IAM**: Handwriting
- **SROIE**: Receipt OCR
- **RVL-CDIP**: Document classification

### 유용한 라이브러리
- **EasyOCR**: https://github.com/JaidedAI/EasyOCR
- **PaddleOCR**: https://github.com/PaddlePaddle/PaddleOCR
- **MMOCR**: https://github.com/open-mmlab/mmocr
- **Tesseract**: https://github.com/tesseract-ocr/tesseract
- **TrOCR**: https://huggingface.co/docs/transformers/model_doc/trocr

### 연구 리소스
- **Papers with Code - OCR**: https://paperswithcode.com/task/optical-character-recognition
- **Awesome OCR**: https://github.com/kba/awesome-ocr
- **OCR Datasets**: https://github.com/cs-chan/Total-Text-Dataset

---

## 8. 요약

### 핵심 포인트
1. **OCR = Detection + Recognition** 두 단계로 구성
2. **시대별 발전**: Rule-based → ML → DL → Attention → Transformer
3. **실무 Top 3**: EasyOCR, PaddleOCR, Tesseract
4. **최신 SOTA**: TrOCR, ABINet, PARSeq, Donut
5. **도메인별 특화**: Scene/Document/Handwriting 각각 최적 모델 존재