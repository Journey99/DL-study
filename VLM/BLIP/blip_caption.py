'''
BLIP Image Captioning

핵심 개념
- vision encoder (ViT)
- text encoder/decoder (transformer)
- end-to-end training with cross-entropy loss and optional CIDEr optimization

'''

import torch
from transformers import BlipForConditionalGeneration, BlipProcessor
from PIL import Image

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

"""BLIP Image Captioning 모델을 사용하여 이미지에 대한 캡션을 생성하는 예제 코드."""

# 모델과 프로세서 로드
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# 이미지 로드
image = Image.open("test.jpg").convert("RGB")

# 전처리
inputs = processor(image, return_tensors="pt").to(device)

# 추론
out = model.generate(**inputs, max_new_tokens=30)
caption = processor.decode(out[0], skip_special_tokens=True)

print("Generated Caption:", caption)
