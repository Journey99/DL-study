from ultralytics import YOLO

# train
model = YOLO("yolo11n.pt")

results = model.train(data="/dataset/aquarium_pretrain/data.yaml",
                      epochs=1000, imgsz=640, batch=16, device=0, workers=2,
                      project="/trainresult/")


# inference
model = YOLO("best.pt")

metrics = model.val(
    data = "dataset/data.yaml",
    split = "test",
    imgsz = 640,
    iou = 0.6
)

print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)

for i, cls_map in enumerate(metrics.box.maps):
    print(f"Class {i}: mAP50-95 = {cls_map:.4f}")
