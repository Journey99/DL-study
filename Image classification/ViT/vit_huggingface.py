# 사전학습된 vit 모델로 이미지 분류 (예제)

from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
import requests

# 1. 이미지 불러오기
url = "https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 2. 이미지 전처리 (ViT 입력 형태로)
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
inputs = processor(images=image, return_tensors="pt")

# 3. 모델 로드 (사전학습 + 분류 헤드 포함)
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

# 4. 예측
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()

# 5. 라벨 확인
print("예측된 클래스 ID:", predicted_class)
print("라벨:", model.config.id2label[predicted_class])
