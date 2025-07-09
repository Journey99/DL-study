# huggingface를 통해 swintransformer 사용 예시

from transformers import SwinForImageClassification, SwinFeatureExtractor
from PIL import Image
import torch
import requests

# 1. 이미지 로딩
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/image_classification.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 2. feature extractor
extractor = SwinFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
inputs = extractor(images=image, return_tensors="pt")

# 3. 모델 불러오기
model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
model.eval()

# 4. 예측
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

# 5. 클래스 확인
print("Predicted class:", model.config.id2label[predicted_class_idx])
