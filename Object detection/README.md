# 📚 객체 탐지(Object Detection) 대표 모델 총정리

이 문서는 객체 탐지에 사용되는 대표적인 모델들의 발전 흐름과 구조, 성능, 사용 가능한 코드/라이브러리 정보를 포함한 정리입니다.

---

## 🧭 객체 탐지 모델 타임라인 요약

| 시대 | 주요 모델 | 비고 |
|------|-----------|------|
| 2014~2015 | R-CNN, Fast R-CNN, Faster R-CNN | Region-based 2단계 탐지 |
| 2016~2018 | YOLOv1~v3, SSD, RetinaNet | 1단계 탐지(실시간 가능) |
| 2019~2020 | YOLOv4, EfficientDet, CenterNet | 경량화 + 정확도 향상 |
| 2021~2023 | YOLOv5~v8, PP-YOLOE, DETR | Transformer 기반·End-to-End |
| 2024~현재 | YOLO-NAS, YOLO-World, DINO-DETR | 대규모 데이터·멀티도메인 |

---

## 📊 대표 모델 비교표

| 모델명 | 발표연도 | 구조 특징 | 장점 | 한계 | mAP@COCO | FPS |
|--------|----------|-----------|------|------|----------|-----|
| **R-CNN** | 2014 | Selective Search + CNN | 높은 정확도 | 느림, 2단계 학습 | 53.3 | 0.05 |
| **Fast R-CNN** | 2015 | RoI Pooling + 단일 CNN | 학습 속도 ↑ | 여전히 제안영역 필요 | 66.9 | 0.5 |
| **Faster R-CNN** | 2015 | RPN + RoI Pooling | 제안영역 생성 자동화 | 실시간 불가 | 69.9 | 7 |
| **YOLOv1** | 2016 | Grid 기반 1단계 탐지 | 매우 빠름 | 작은 객체 어려움 | 63.4 | 45 |
| **SSD** | 2016 | Multi-scale feature map | 속도·정확도 균형 | 작은 객체 한계 | 74.3 | 46 |
| **RetinaNet** | 2017 | FPN + Focal Loss | 클래스 불균형 해결 | YOLO보다 느림 | 79.5 | 11 |
| **YOLOv3** | 2018 | Darknet-53 + FPN | 다양한 크기 탐지 | 무거움 | 80.0 | 30 |
| **EfficientDet-D0** | 2020 | EfficientNet backbone | 경량·모바일 친화 | 구현 복잡 | 77.2 | 40 |
| **DETR** | 2020 | Transformer + bipartite matching | End-to-End | 학습 느림 | 76.1 | 28 |
| **YOLOv8n** | 2023 | Anchor-free, Conv + CSP | 매우 빠름, 쉬운 사용 | 고해상도 시 리소스↑ | 78.9 | 120 |
| **YOLO-NAS** | 2024 | NAS 최적화 구조 | 속도·정확도 최고 수준 | 모델 크기 큼 | 80+ | 130 |

---

## 🔧 모델별 코드 사용법

| 모델 | 라이브러리 | 사용 방법 |
|------|------------|-----------|
| **Faster R-CNN** | Detectron2 | ```python<br>from detectron2.engine import DefaultPredictor``` |
| **YOLOv8** | ultralytics | `YOLO('yolov8n.pt')` |
| **YOLO-NAS** | super-gradients | `YOLO_NAS("yolo_nas_s")` |
| **EfficientDet** | timm | `timm.create_model("tf_efficientdet_d0", pretrained=True)` |
| **DETR** | transformers | `AutoModel.from_pretrained("facebook/detr-resnet-50")` |

---

## 🔍 주요 개념 요약

| 개념 | 설명 |
|------|------|
| **Two-stage Detection** | Region proposal + Classification (R-CNN 계열) |
| **One-stage Detection** | 바로 예측 (YOLO, SSD, RetinaNet) |
| **Anchor Box** | 사전 정의된 박스 비율·크기 |
| **FPN (Feature Pyramid Network)** | 멀티스케일 특징 활용 |
| **Focal Loss** | 클래스 불균형 완화 |
| **Transformer in Detection** | DETR, DINO-DETR: End-to-End 학습 |
| **Anchor-free Detection** | 중심점 기반 탐지 (YOLOv8, FCOS) |
| **Region Proposal** | 후보 영역의 좌표 정보 |
| **RoI(region of interest)** | region proposal을 이용해 feature map에서 해당 영역을 잘라낸 것. roi pooling을 거쳐 고정된 크기의 특징맴으로 변환 |
---

## 🔚 요약 정리

- **2014~2016**: R-CNN → Fast R-CNN → Faster R-CNN (정확도 ↑, 속도 ↑)
- **2016~2020**: YOLO, SSD, RetinaNet (실시간, 효율성)
- **2020 이후**: Transformer 기반 모델(End-to-End), NAS 기반 최적화
- **현재 추세**: YOLO 계열이 실시간·산업 적용 강세, DETR 계열은 연구·고정밀 영역

---

## ✅ 참고

- Detectron2 Docs: https://detectron2.readthedocs.io
- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
- SuperGradients YOLO-NAS: https://github.com/Deci-AI/super-gradients
- HuggingFace DETR: https://huggingface.co/facebook/detr-resnet-50
