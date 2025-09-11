# YOLO Series 총정리 

## 목차
1. YOLOv1 (2016)  
2. YOLOv2 / YOLO9000 (2016)  
3. YOLOv3 (2018)  
4. YOLOv4 ~ YOLOv7 (2020~2022) — 발전/최적화 시리즈  
5. YOLOv8 (2023) — Ultralytics 통합 API  
6. YOLO-NAS (2023) — NAS 기반 최적화 (Deci) 
7. YOLO-World (2024) — Open-Vocabulary (CVPR 2024) 
8. YOLOv10 (2024) — End-to-End / NMS-free 연구 논문 
9. YOLOv11 (2024) — Ultralytics / 아키텍처 개선 문서 
10. YOLOv13 (2025) — Hypergraph 기반 고차 상관성 강화 (연구) 


## 1) YOLOv1 (2016)
- **한줄 요약**: 이미지를 S×S grid로 나누고 각 셀에서 bbox + 클래스 확률을 예측 — 객체 탐지를 회귀 문제로 단순화하여 실시간 처리 가능하게 함.
- **주요 특징**
  - proposal-free, single-pass inference
  - 빠른 속도 장점, 작은 객체/클러스터된 객체에 약함
- **발전된 기술**: 통합 회귀(loss: 분류 + 좌표) 관점 도입
- **구조**
  - 단일 CNN (24 conv + 2 FC in original)
  - Grid 기반 출력: 각 cell이 B bboxes와 C class 확률 예측


## 2) YOLOv2 / YOLO9000 (2016)
- **한줄 요약**: anchor 도입, multi-scale 학습, 더 강력한 backbone(Darknet-19) 적용 → 정확도/속도 균형 개선.
- **주요 특징**
  - **Anchor boxes** 적용 (SSD 스타일)  
  - **Dimension clustering**로 anchor 초기화
  - **Multi-scale training** (학습 중 입력 해상도 변화)
  - YOLO9000: 대규모 계층적 라벨 학습으로 더 많은 클래스(weakly-supervised) 대응
- **구조**
  - Backbone: Darknet-19
  - Neck/Head: conv 기반 detection heads (anchor 사용)


## 3) YOLOv3 (2018)
- **한줄 요약**: Darknet-53 backbone + multi-scale detection (3-level) 도입으로 작은 물체 성능 개선.
- **주요 특징**
  - **Darknet-53** (잔차 블록 포함) 백본
  - **다중 스케일(3-level) 예측**: 작은 / 중간 / 큰 물체에 대응
  - logistic class predictions, binary cross-entropy 등 개선된 학습 세팅
- **구조**
  - Backbone: Darknet-53
  - Neck: upsampling + concatenation (간단한 FPN 스타일)
  - Head: 각 스케일마다 bbox+conf+class 예측


## 4) YOLOv4 ~ YOLOv7 (2020 ~ 2022)
- **한줄 요약**: 여러 최적화 (Bag-of-freebies / Bag-of-specials), CSP 구조·SPP·PAN 등으로 정확도와 효율을 함께 끌어올림.
- **주요 특징 (요약)**
  - **CSPDarknet** (Cross Stage Partial) 백본(gradient/compute 효율 개선)
  - **SPP (Spatial Pyramid Pooling)**, **PAN (Path Aggregation Network)** 등 neck 설계
  - 다양한 학습 트릭(데이터 증강, 백본/neck의 상세 튜닝)
  - YOLOv7: 설계 최적화로 SOTA 근접 + multi-task 확장(예: pose/seg)
- **구조**
  - Backbone: CSP variants
  - Neck: SPP / PAN / 기타 fusion modules
  - Head: multi-scale heads (anchor or anchor-free depending on variant)


## 5) YOLOv8 (2023) — Ultralytics
- **한줄 요약**: **Anchor-free** 접근, 통합 API(ultralytics)로 사용 편의성 향상, 멀티태스크(검출/분류/세그멘테이션/트래킹) 지원.
- **주요 특징**
  - anchor-free box prediction
  - 통합된 사용자 API (train/val/infer) — 실무 프로토타입 빠르게 제작 가능
  - 다양한 사전학습 체크포인트 제공(유연한 스케일: n/s/m/l)
- **구조**
  - Backbone: Ultralytics custom (경량화에 초점)
  - Neck: PAN-like
  - Head: anchor-free single-pass head
- **실무 노트**
  - 초보자/산업팀에서 많이 채택 — 사용성과 문서화 강점.



## 6) YOLO-NAS (2023) — Deci (NAS 기반) 
- **한줄 요약** : **Neural Architecture Search(NAS)** 를 통해 속도/정확도/모델 크기 트레이드오프를 자동 최적화한 YOLO 계열 모델.
- **주요 특징 / 발전 기술**
  - NAS로 backbone/neck/head 조합을 자동 탐색 → 효율성 최적화
  - Quantization-Aware Training(실제 하드웨어 배포 고려)
  - 실제 엣지 성능(지연/메모리)까지 고려한 모델 설계
- **구조(일반적)**  
  - Backbone/Neck/Head를 NAS로 탐색·설계한 family (nano → large)
- **실무 노트**
  - 엣지·임베디드 배포에 유리. Deci의 구현/튜토리얼(예: super-gradients)으로 실무 적용 가능.



## 7) YOLO-World (2024) — Open-Vocabulary YOLO (CVPR 2024)
- **한줄 요약**: YOLO 계열에 **vision-language(open-vocabulary)** 학습을 도입해 사전 정의된 클래스셋 밖 객체에 대해서도 감지 가능한 실시간 OVD(오픈보캐뷸러리 디텍션) 구현.
- **주요 특징 / 발전 기술**
  - Vision–language pretraining + region-text contrastive loss
  - Open-vocabulary (zero-shot) 탐지 성능 향상
  - 실시간 추론을 유지하면서 OVD 목표 달성
- **구조**
  - 기존 YOLO backbone에 region-to-text 모듈 / language embedding fusion
- **실무 노트**
  - 라벨링 비용 문제, 새로운 클래스 탐지 등이 요구되는 어플리케이션(예: 리테일, 로봇)에서 유용.



## 8) YOLOv10 (2024) — End-to-End / NMS-free 제안 (논문)
- **한줄 요약**: 기존 YOLO의 post-processing(NMS)을 제거(또는 대체)하여 **end-to-end, NMS-free 학습/추론**을 목표로 하고, 아키텍처 전반을 효율-정확도 관점에서 재설계.
- **주요 특징 / 발전 기술**
  - **NMS-free / dual assignment** 기반 학습 전략 (consistent dual assignments)
  - 모델 아키텍처 전반(효율성 관점) 재설계 — latency 대폭 개선
- **구조**
  - 효율성 중심 재설계된 backbone + NMS-free head(dual assignment 기반)
- **실무 노트**
  - NMS 제거로 파이프라인 단순화 가능(특히 multi-threaded/low-latency 환경).



## 9) YOLOv11 (2024) — Ultralytics / community 문서화 및 개선 
- **한줄 요약**: Ultralytics가 문서·도구 측면에서 통합한 최신 YOLO 배포판(아키텍처 개선·학습 파이프라인 고도화 포함).
- **주요 특징**
  - C3k2, SPPF, Parallel Spatial Attention 등 모듈 도입(문서 기준)
  - 멀티태스크 확장(Detection, Segmentation, Pose 등) 및 다양한 모델 스케일 제공
- **구조**
  - Ultralytics 가이드라인에 따른 Backbone/Neck 개선(실무 적용용 튜닝 포함)


## 10) YOLOv13 (2025) — Hypergraph 기반 고차 상관성 강화 (연구) 
- **한줄 요약**: 지역 기반 정보(Convolution)와 기존 self-attention의 한계를 보완하기 위해 **하이퍼그래프 기반의 고차 상관성 모듈(HyperACE)** 를 도입, 글로벌 다대다 관계를 모델링하여 복잡한 장면에서 성능 향상 시도.
- **주요 특징 / 발전 기술**
  - 고차 상관성(Hypergraph) 모듈로 multi-to-multi 관계 포착
  - 경량 설계를 유지하면서 글로벌 문맥 보강
- **구조**
  - HyperACE 모듈이 neck/head에 삽입되어 feature 간 고차 상관성 강화


## 비교 요약 
- **속도 중심(엣지)**: YOLO-NAS, YOLOv10 (latency 최적화)  
- **통합 & 사용성**: YOLOv8 / YOLOv11 (Ultralytics)  
- **오픈보캐뷸러리 / 제너럴리제이션**: YOLO-World  
- **연구 전선(최신)**: YOLOv13 (Hypergraph 등 새로운 모듈)

## 공통 핵심 개념

### 1. Grid 기반 detection
- 개념: YOLOv1에서는 이미지를 S×S 그리드로 나누고, 각 셀이 물체의 중심을 책임지도록 설계.
- 작동 방식: 각 그리드 셀은 "이 셀 안에 물체가 있다/없다"를 판단하고, 있으면 bounding box와 클래스 확률을 동시에 예측.
- 장점: 단일 forward pass로 빠른 예측 가능.
- 한계: 작은 물체, 인접한 물체에 약함 → 이후 Anchor box, FPN 도입으로 개선.

### 2. Anchor Box
![alt text](image.png)
- 등장: yolov2부터
- 개념 : 다양한 크기와 비율의 박스를 미리 정의(anchor)해 두고, 실제 박스를 anchor를 기준으로 보정하는 방식
  - 모델의 예측 단계에서 사용할 기준이 되는 박스를 제공하는 것이며, 다양한 스케일의 객체를 탐지할 수 있게 한다
  - 즉, 바운딩 박스를 예측하고 조정하며 Objectness score를 계산하기 위하여 만들어지는 Object가 있을 candidate region이다
- 개선 과정
  - YOLOv2: k-means 클러스터링으로 Anchor 자동 학습.
  - YOLOv3: 3개의 다른 스케일에서 Anchor 사용 (multi-scale).
  - YOLOv4 이후: PANet/FPN과 결합하여 작은 물체 성능 강화.