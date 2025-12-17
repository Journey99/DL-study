import os
import glob
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# IoU 계산
def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0

# YOLO label -> xyxy
def yolo_to_xyxy(label, img_w, img_h):
    cls, cx, cy, w, h = label
    cx, cy = cx * img_w, cy * img_h
    w, h = w * img_w, h * img_h

    return int(cls), [
        cx - w / 2,
        cy - h / 2,
        cx + w / 2,
        cy + h / 2
    ]

# GT load
def load_gt(label_path, img_w, img_h):
    gts = []
    if not os.path.exists(label_path):
        return gts

    with open(label_path) as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            gts.append(yolo_to_xyxy(parts, img_w, img_h))

    return gts


def collect_predictions(
    model,
    img_dir,
    label_dir,
    conf_thresh=0.25
):
    preds = defaultdict(list)
    gts = defaultdict(list)

    image_paths = glob.glob(os.path.join(img_dir, "*.jpg"))

    for img_path in image_paths:
        label_path = os.path.join(
            label_dir,
            os.path.basename(img_path).replace(".jpg", ".txt")
        )

        result = model(img_path, conf=conf_thresh)[0]
        img_h, img_w = result.orig_shape

        # GT
        gt_boxes = load_gt(label_path, img_w, img_h)
        for cls, box in gt_boxes:
            gts[cls].append({
                "image": img_path,
                "box": box,
                "used": False
            })

        # Prediction
        if result.boxes is not None:
            for box in result.boxes:
                cls = int(box.cls[0])
                preds[cls].append({
                    "image": img_path,
                    "box": box.xyxy[0].cpu().numpy().tolist(),
                    "conf": float(box.conf[0])
                })

    return preds, gts

# AP
def compute_ap(recalls, precisions):
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])

    idx = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum(
        (recalls[idx + 1] - recalls[idx]) * precisions[idx + 1]
    )
    return ap

def evaluate_class(preds, gts, iou_thresh=0.5):
    preds = sorted(preds, key=lambda x: x["conf"], reverse=True)

    TP = np.zeros(len(preds))
    FP = np.zeros(len(preds))

    total_gt = len(gts)

    for i, pred in enumerate(preds):
        matched = False
        for gt in gts:
            if gt["image"] != pred["image"] or gt["used"]:
                continue

            if compute_iou(pred["box"], gt["box"]) >= iou_thresh:
                TP[i] = 1
                gt["used"] = True
                matched = True
                break

        if not matched:
            FP[i] = 1

    tp_cum = np.cumsum(TP)
    fp_cum = np.cumsum(FP)

    recalls = tp_cum / (total_gt + 1e-6)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-6)

    return compute_ap(recalls, precisions)

# mAP
def evaluate_map(
    model,
    img_dir,
    label_dir,
    conf_thresh=0.25,
    iou_thresh=0.5
):
    preds, gts = collect_predictions(
        model,
        img_dir,
        label_dir,
        conf_thresh
    )

    ap_per_class = {}

    for cls in gts.keys():
        if cls not in preds:
            ap_per_class[cls] = 0.0
            continue

        # GT used flag 초기화
        for gt in gts[cls]:
            gt["used"] = False

        ap = evaluate_class(
            preds[cls],
            gts[cls],
            iou_thresh
        )
        ap_per_class[cls] = ap

    mAP = np.mean(list(ap_per_class.values()))
    return mAP, ap_per_class


if __name__ == "__main__":
    MODEL_PATH = "best.pt"
    IMG_DIR = "datasets/test/images"
    LABEL_DIR = "datasets/test/labels"

    model = YOLO(MODEL_PATH)

    mAP, ap_per_class = evaluate_map(
        model,
        IMG_DIR,
        LABEL_DIR,
        conf_thresh=0.25,
        iou_thresh=0.5
    )

    print("===== mAP@0.5 =====")
    for cls, ap in ap_per_class.items():
        print(f"Class {cls}: AP = {ap:.4f}")

    print(f"\nOverall mAP@0.5: {mAP:.4f}")
