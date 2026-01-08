# BLIP VQA (Visual Question Answering)

from transformers import BlipForQuestionAnswering, BlipProcessor
import torch
from PIL import Image     

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

"""BLIP VQA 모델을 사용하여 이미지와 질문에 대한 답변을 생성하는 예제 코드."""
# 모델과 프로세서 로드
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)

# 이미지 로드
image = Image.open("test.jpg").convert("RGB")   

# 질문 정의
question = "how many people are there?"

# 전처리
inputs = processor(image, question, return_tensors="pt").to(device)    

out = model.generate(**inputs, max_new_tokens=10)
answer = processor.decode(out[0], skip_special_tokens=True)

print("Question:", question)    
print("Answer:", answer)    