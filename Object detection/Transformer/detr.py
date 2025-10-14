# detr huggingface 추론 예시
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import requests
import torch

# 이미지 불러오기
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/detr_image.png"
image = Image.open(requests.get(url, stream=True).raw)

# Processor & 모델 불러오기 (ResNet-50 Backbone)
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# 입력 변환
inputs = processor(images=image, return_tensors="pt")

# 모델 추론
outputs = model(**inputs)

# 결과 디코딩 (threshold=0.9 이상인 객체만)
target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

# 출력
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    if score > 0.9:
        box = [round(i, 2) for i in box.tolist()]
        print(f"Detected {model.config.id2label[label.item()]} with confidence {round(score.item(), 3)} at location {box}")