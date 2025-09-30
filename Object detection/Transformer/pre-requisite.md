# 📘 Pre-requisite: Transformer 기반 Object Detection 이해를 위한 사전 지식

Transformer 기반 객체 탐지 모델(DETR, Deformable DETR, DAB-DETR, DINO 등)을 이해하기 전에 반드시 알고 있어야 할 기본 개념들을 정리한 것이다.


## 1. Transformer 기본 구조
- **Self-Attention**  
  입력 feature들이 서로 관계를 맺으며 global context를 학습.  
- **Encoder–Decoder 구조**  
  - Encoder: 이미지 전체 feature를 인코딩  
  - Decoder: Object Query를 입력받아 객체 위치와 클래스를 디코딩  
- **Position Embedding**  
  CNN과 달리 순서 정보가 없기 때문에 위치 정보를 sine/cosine embedding으로 추가  



## 2. Object Query
- DETR 계열의 핵심 아이디어  
- 학습 가능한 embedding (vector)으로, 디코더에 입력됨  
- 디코더는 각 query를 반복 업데이트하여 **"어떤 객체가 어디에 있는지"**를 예측  



## 3. Bipartite Matching (Hungarian Algorithm)
- 기존 CNN 기반 detection은 NMS(Non-Maximum Suppression) 사용  
- DETR은 **set prediction** 방식을 채택  
- 예측 박스와 GT 박스를 **1:1 매칭** → 중복 탐지 제거, 더 깔끔한 학습 가능  


## 4. Loss Functions
- **Classification Loss**: Cross-Entropy Loss (클래스 분류)  
- **Localization Loss**: L1 Loss + GIoU/DIoU/CIoU Loss (박스 좌표 회귀)  
- **Set-based Loss**: Hungarian matching으로 매칭된 pair에만 적용  


## 5. Multi-scale Feature Representation
- 객체 크기 다양성 문제 해결  
- **FPN (Feature Pyramid Network)**: CNN 기반 multi-scale feature 사용  
- **Deformable DETR**: multi-scale feature map에서 샘플링하여 attention  


## 6. Deformable Attention
- Standard attention은 모든 픽셀에 대해 attention → 계산량 ↑  
- Deformable Attention은 **일부 샘플링 포인트만 선택적으로 참조**  
- → 연산 효율 ↑, 고해상도 이미지 처리 가능  


## 7. Query Denoising
- 학습 초기에 수렴 속도 문제 해결을 위해 등장 (DN-DETR, DINO 등)  
- GT box/label에 noise를 추가한 query를 학습시켜 모델이 더 안정적으로 학습  

## 8. Dynamic Anchor Box (DAB-DETR)
- 기존 DETR의 object query는 단순 vector  
- DAB-DETR에서는 query를 **위치 정보가 포함된 anchor 형태**로 초기화  
- 학습 속도와 안정성 개선  

## 9. Anchor-free Detection
- 기존 (YOLO, SSD 등) → Anchor Box 기반 (미리 정의된 박스와 매칭)  
- DETR 계열 → **Anchor-free** → 좌표를 직접 회귀하여 예측  

## 10. Vision-Language 확장
- 최신 Transformer 기반 모델들은 텍스트와 결합  
  - **Grounding-DINO**: 텍스트 조건 기반 객체 탐지  
  - **GLIP**: Object detection + Text grounding 통합 학습  

---

## ✅ 요약
- **Transformer 기본기**: attention, encoder-decoder, position embedding  
- **DETR의 핵심**: object query, bipartite matching, set-based loss  
- **성능 개선 요소**: multi-scale feature, deformable attention, query denoising, dynamic anchor  
- **최신 추세**: Vision-Language 융합  

