'''
[BLIP-2 핵심 구조]
iamge -> vision encoder (e.g., ViT, CLIP) --> frozen
          |
          v
       Q-Former (frozen)
          |
          v
       language model (e.g., GPT2) --> fine-tuned for captioning

: 학습 대상은 Q-Former + projection

[중요 포인트]
- ViT 출력 전체를 LLM에 넣지 않음
- Q-Former가 요약된 시각 토큰만 뽑아냄
- LLM은 완전히 frozen
'''

from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image

# ============================================
# 1. 디바이스 설정
# ============================================
def get_device():
    """
    사용 가능한 최적의 디바이스 반환
    """
    if torch.cuda.is_available():
        return 'cuda', torch.float16
    elif torch.backends.mps.is_available():
        return 'mps', torch.float32
    else:
        return 'cpu', torch.float32

# ============================================
# 2. BLIP-2 모델 로드
# ============================================
def load_blip2_model(model_name="Salesforce/blip2-flan-t5-xl"):
    """
    BLIP-2 모델과 프로세서 로드
    
    Available models:
        - blip2-opt-2.7b: 작고 빠름
        - blip2-opt-6.7b: 중간 크기
        - blip2-flan-t5-xl: 높은 성능 (추천)
        - blip2-flan-t5-xxl: 최고 성능 (무거움)
    
    Returns:
        processor, model, device
    """
    device, dtype = get_device()
    
    print(f"Loading BLIP-2: {model_name}")
    print(f"Device: {device}, Dtype: {dtype}\n")
    
    # Processor 로드
    processor = Blip2Processor.from_pretrained(model_name)
    
    # Model 로드
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype
    ).to(device)
    
    print("✓ Model loaded successfully\n")
    
    return processor, model, device

# ============================================
# 3. Image Captioning
# ============================================
def generate_caption(processor, model, device, image_path, max_length=50):
    """
    이미지에 대한 캡션 생성
    
    Args:
        image_path: 이미지 파일 경로 또는 PIL Image
        max_length: 생성할 최대 토큰 수
    
    Returns:
        caption: 생성된 캡션 (str)
    """
    # 이미지 로드
    if isinstance(image_path, str):
        image = Image.open(image_path).convert("RGB")
    else:
        image = image_path
    
    # 전처리
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    # 캡션 생성
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_length)
    
    # 디코딩
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    
    return caption

# ============================================
# 4. Visual Question Answering (VQA)
# ============================================
def answer_question(processor, model, device, image_path, question, max_length=30):
    """
    이미지에 대한 질문에 답변
    
    Args:
        image_path: 이미지 파일 경로 또는 PIL Image
        question: 질문 (str)
        max_length: 답변 최대 길이
    
    Returns:
        answer: 답변 (str)
    """
    # 이미지 로드
    if isinstance(image_path, str):
        image = Image.open(image_path).convert("RGB")
    else:
        image = image_path
    
    # 전처리 (이미지 + 텍스트)
    inputs = processor(image, question, return_tensors="pt").to(device)
    
    # 답변 생성
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_length)
    
    # 디코딩
    answer = processor.decode(outputs[0], skip_special_tokens=True)
    
    return answer

# ============================================
# 5. 배치 처리
# ============================================
def batch_caption(processor, model, device, image_paths, max_length=50):
    """
    여러 이미지를 동시에 처리
    
    Args:
        image_paths: 이미지 경로 리스트
    
    Returns:
        captions: 캡션 리스트
    """
    images = [Image.open(path).convert("RGB") for path in image_paths]
    
    # 배치 전처리
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    
    # 배치 생성
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_length)
    
    # 디코딩
    captions = [processor.decode(out, skip_special_tokens=True) for out in outputs]
    
    return captions

# ============================================
# 6. 고급 생성 옵션
# ============================================
def generate_with_options(
    processor, 
    model, 
    device, 
    image_path, 
    prompt=None,
    num_beams=5,
    temperature=1.0,
    top_p=0.9,
    max_length=50
):
    """
    다양한 생성 옵션으로 캡션 생성
    
    Args:
        prompt: 조건부 생성 (e.g., "A photo of")
        num_beams: Beam search width (높을수록 품질 향상, 느림)
        temperature: 다양성 조절 (낮을수록 보수적)
        top_p: Nucleus sampling threshold
    
    Returns:
        caption: 생성된 캡션
    """
    # 이미지 로드
    if isinstance(image_path, str):
        image = Image.open(image_path).convert("RGB")
    else:
        image = image_path
    
    # 전처리 (prompt가 있으면 포함)
    if prompt:
        inputs = processor(image, prompt, return_tensors="pt").to(device)
    else:
        inputs = processor(images=image, return_tensors="pt").to(device)
    
    # 고급 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,  # temperature > 0이면 sampling
        )
    
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    
    return caption

# ============================================
# 7. 다양한 질문 타입
# ============================================
def ask_multiple_questions(processor, model, device, image_path, questions):
    """
    하나의 이미지에 대해 여러 질문
    
    Args:
        questions: 질문 리스트
    
    Returns:
        qa_pairs: [(question, answer), ...] 리스트
    """
    image = Image.open(image_path).convert("RGB") if isinstance(image_path, str) else image_path
    
    qa_pairs = []
    
    for question in questions:
        answer = answer_question(processor, model, device, image, question)
        qa_pairs.append((question, answer))
    
    return qa_pairs

# ============================================
# 8. 메인 실행 예시
# ============================================
def main():
    # 모델 로드
    processor, model, device = load_blip2_model("Salesforce/blip2-flan-t5-xl")
    
    # 테스트 이미지
    image_path = "test.jpeg"
    
    print("="*60)
    print("1. Image Captioning")
    print("="*60)
    
    caption = generate_caption(processor, model, device, image_path)
    print(f"Caption: {caption}\n")
    
    print("="*60)
    print("2. Visual Question Answering (VQA)")
    print("="*60)
    
    question = "What is the person doing?"
    answer = answer_question(processor, model, device, image_path, question)
    print(f"Q: {question}")
    print(f"A: {answer}\n")
    
    print("="*60)
    print("3. Multiple Questions")
    print("="*60)
    
    questions = [
        "What is in the image?",
        "What color is the main object?",
        "Where is this photo taken?",
        "What time of day is it?"
    ]
    
    qa_pairs = ask_multiple_questions(processor, model, device, image_path, questions)
    
    for q, a in qa_pairs:
        print(f"Q: {q}")
        print(f"A: {a}\n")
    
    print("="*60)
    print("4. Advanced Generation (with beam search)")
    print("="*60)
    
    caption_advanced = generate_with_options(
        processor, model, device, image_path,
        num_beams=5,
        max_length=100
    )
    print(f"Detailed Caption: {caption_advanced}\n")

if __name__ == "__main__":
    main()