'''
https://github.com/facebookresearch/segment-anything
'''

# 추론 실습
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamPredictor

# 이미지 로드
image = cv2.imread('test.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 모델 로드
sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

# 이미지 세팅 : 이 시점에서 image embedding 생성
predictor.set_image(image)

# point prompt 기반 segmentation
input_point = np.array([[500, 375]])
input_label = np.array([1])  # 1: foreground

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True
)

# 결과 시각화
def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255])
    h, w = mask.shape
    mask_image = mask[:, :, None] * color[None, None, :]
    ax.imshow(mask_image)

plt.figure(figsize=(10, 10))
plt.imshow(image)
show_mask(masks[0], plt.gca())
plt.axis('off')
plt.show()