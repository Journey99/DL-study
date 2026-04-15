import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def show_mask(mask, ax, color=np.array([30 / 255, 144 / 255, 255 / 255, 0.45])):
    mask = mask.astype(np.uint8)
    h, w = mask.shape
    mask_image = np.zeros((h, w, 4), dtype=np.float32)
    mask_image[..., :4] = color
    mask_image[..., 3] = color[3] * mask
    ax.imshow(mask_image)


def show_box(box, ax, edge_color="yellow"):
    x0, y0, x1, y1 = box
    ax.add_patch(
        plt.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            edgecolor=edge_color,
            facecolor=(0, 0, 0, 0),
            linewidth=2,
        )
    )


# 1) 이미지 로드
image_path = "test.jpg"
text_prompt = "person"  # 예: "yellow school bus", "dog", "red car"
image = Image.open(image_path).convert("RGB")

# 2) SAM3 모델/프로세서 로드
model = build_sam3_image_model()
processor = Sam3Processor(model)

# 3) 이미지 세션 시작 후 텍스트 프롬프트 입력
state = processor.set_image(image)
output = processor.set_text_prompt(state=state, prompt=text_prompt)

# output dict: "masks", "boxes", "scores"
masks = output["masks"]
boxes = output["boxes"]
scores = output["scores"]

if len(scores) == 0:
    raise RuntimeError(f"No result for prompt: {text_prompt}")

# 4) 최고 점수 결과 시각화
best_idx = int(np.argmax(scores))
best_mask = masks[best_idx]
best_box = boxes[best_idx]
best_score = float(scores[best_idx])

plt.figure(figsize=(10, 10))
plt.imshow(image)
show_mask(best_mask, plt.gca())
show_box(best_box, plt.gca())
plt.title(f'SAM3 prompt="{text_prompt}", score={best_score:.4f}')
plt.axis("off")
plt.tight_layout()
plt.show()