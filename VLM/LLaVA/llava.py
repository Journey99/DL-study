'''
LLaVA 추론 코드
- multimodel LLM : 이미지 이해 + 대화
- GPT-4V 스타일의 비전-언어 능력
'''

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import requests
from io import BytesIO

# ============================================
# 1. 디바이스 설정
# ============================================
def get_device():
    """최적 디바이스 및 dtype 반환"""
    if torch.cuda.is_available():
        return 'cuda', torch.float16
    elif torch.backends.mps.is_available():
        return 'mps', torch.float32
    else:
        return 'cpu', torch.float32

# ============================================
# 2. LLaVA 모델 로드
# ============================================
def load_llava_model(model_name="llava-hf/llava-1.5-7b-hf"):
    """
    LLaVA 모델 로드
    
    Available models:
        - llava-hf/llava-1.5-7b-hf: 기본 모델 (추천)
        - llava-hf/llava-1.5-13b-hf: 큰 모델 (더 좋은 성능)
        - llava-hf/llava-v1.6-mistral-7b-hf: Mistral 기반 (최신)
        - llava-hf/llava-v1.6-vicuna-13b-hf: Vicuna 기반
    
    Returns:
        processor, model, device
    """
    device, dtype = get_device()
    
    print(f"Loading LLaVA: {model_name}")
    print(f"Device: {device}, Dtype: {dtype}")
    print("This may take a few minutes...\n")
    
    # Processor 로드
    processor = AutoProcessor.from_pretrained(model_name)
    
    # Model 로드
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    
    print("✓ Model loaded successfully\n")
    
    return processor, model, device

# ============================================
# 3. 이미지 로드
# ============================================
def load_image(image_source):
    """
    이미지 로드 (파일 경로, URL, PIL Image 모두 지원)
    
    Args:
        image_source: 파일 경로(str), URL(str), 또는 PIL Image
    
    Returns:
        PIL Image
    """
    if isinstance(image_source, str):
        # URL인 경우
        if image_source.startswith(('http://', 'https://')):
            response = requests.get(image_source)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        # 파일 경로인 경우
        else:
            image = Image.open(image_source).convert('RGB')
    else:
        # 이미 PIL Image인 경우
        image = image_source
    
    return image

# ============================================
# 4. 기본 대화 (Single-turn)
# ============================================
def chat(processor, model, device, image_source, prompt, max_new_tokens=512):
    """
    이미지에 대한 단일 질문-답변
    
    Args:
        image_source: 이미지 (경로, URL, PIL Image)
        prompt: 사용자 질문/명령
        max_new_tokens: 생성할 최대 토큰 수
    
    Returns:
        response: 모델의 답변
    """
    # 이미지 로드
    image = load_image(image_source)
    
    # LLaVA는 특정 대화 형식 필요
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    
    # 프롬프트 생성
    prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    # 입력 전처리
    inputs = processor(images=image, text=prompt_text, return_tensors="pt").to(device)
    
    # 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding (일관된 답변)
        )
    
    # 디코딩 (입력 프롬프트 제외)
    generated_text = processor.decode(outputs[0], skip_special_tokens=True)
    
    # 답변 부분만 추출
    response = generated_text.split("ASSISTANT:")[-1].strip()
    
    return response

# ============================================
# 5. 다양한 생성 옵션
# ============================================
def chat_with_options(
    processor,
    model,
    device,
    image_source,
    prompt,
    temperature=0.2,
    top_p=0.9,
    max_new_tokens=512,
    do_sample=True
):
    """
    고급 생성 옵션으로 대화
    
    Args:
        temperature: 다양성 조절 (0에 가까울수록 일관적)
        top_p: Nucleus sampling threshold
        do_sample: True면 sampling, False면 greedy
    """
    image = load_image(image_source)
    
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    
    prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
        )
    
    generated_text = processor.decode(outputs[0], skip_special_tokens=True)
    response = generated_text.split("ASSISTANT:")[-1].strip()
    
    return response

# ============================================
# 6. Multi-turn 대화
# ============================================
class LLaVAChat:
    """
    Multi-turn 대화를 위한 클래스
    """
    def __init__(self, processor, model, device, image_source):
        self.processor = processor
        self.model = model
        self.device = device
        self.image = load_image(image_source)
        self.conversation_history = []
    
    def add_message(self, role, content):
        """대화 히스토리에 메시지 추가"""
        self.conversation_history.append({
            "role": role,
            "content": content
        })
    
    def chat(self, user_message, max_new_tokens=512):
        """
        사용자 메시지를 받아 응답 생성
        
        Args:
            user_message: 사용자의 질문/명령
        
        Returns:
            assistant_response: 모델의 답변
        """
        # 사용자 메시지 추가 (첫 메시지에만 이미지 포함)
        if len(self.conversation_history) == 0:
            content = [
                {"type": "image"},
                {"type": "text", "text": user_message}
            ]
        else:
            content = [{"type": "text", "text": user_message}]
        
        self.add_message("user", content)
        
        # 프롬프트 생성
        prompt_text = self.processor.apply_chat_template(
            self.conversation_history,
            add_generation_prompt=True
        )
        
        # 입력 준비
        inputs = self.processor(
            images=self.image,
            text=prompt_text,
            return_tensors="pt"
        ).to(self.device)
        
        # 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        
        # 디코딩
        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        assistant_response = generated_text.split("ASSISTANT:")[-1].strip()
        
        # 어시스턴트 응답 추가
        self.add_message("assistant", [{"type": "text", "text": assistant_response}])
        
        return assistant_response
    
    def reset(self):
        """대화 히스토리 초기화"""
        self.conversation_history = []

# ============================================
# 7. 배치 처리
# ============================================
def batch_chat(processor, model, device, image_prompts, max_new_tokens=512):
    """
    여러 이미지-프롬프트 쌍을 동시 처리
    
    Args:
        image_prompts: [(image1, prompt1), (image2, prompt2), ...]
    
    Returns:
        responses: 답변 리스트
    """
    responses = []
    
    for image_source, prompt in image_prompts:
        response = chat(processor, model, device, image_source, prompt, max_new_tokens)
        responses.append(response)
    
    return responses

# ============================================
# 8. 다양한 태스크 예제
# ============================================
def image_captioning(processor, model, device, image_source):
    """이미지 캡션 생성"""
    prompt = "Describe this image in detail."
    return chat(processor, model, device, image_source, prompt)

def visual_question_answering(processor, model, device, image_source, question):
    """VQA"""
    return chat(processor, model, device, image_source, question)

def ocr_text_reading(processor, model, device, image_source):
    """이미지 내 텍스트 읽기"""
    prompt = "What text can you see in this image? Please transcribe all visible text."
    return chat(processor, model, device, image_source, prompt)

def object_counting(processor, model, device, image_source, object_name):
    """객체 개수 세기"""
    prompt = f"How many {object_name} are in this image? Count carefully."
    return chat(processor, model, device, image_source, prompt)

def scene_understanding(processor, model, device, image_source):
    """장면 이해"""
    prompt = "What is happening in this image? Describe the scene, activities, and context."
    return chat(processor, model, device, image_source, prompt)

def image_comparison(processor, model, device, image1, image2):
    """
    두 이미지 비교 (연속 대화)
    """
    # 첫 번째 이미지 분석
    chat_session = LLaVAChat(processor, model, device, image1)
    response1 = chat_session.chat("Describe this image.")
    
    # 두 번째 이미지로 새 세션
    chat_session2 = LLaVAChat(processor, model, device, image2)
    response2 = chat_session2.chat("Describe this image.")
    
    return response1, response2

# ============================================
# 9. 메인 실행 예시
# ============================================
def main():
    # 모델 로드
    processor, model, device = load_llava_model("llava-hf/llava-1.5-7b-hf")
    
    # 테스트 이미지
    image_path = "test.jpeg"
    
    print("="*70)
    print("1. Basic Image Captioning")
    print("="*70)
    
    caption = image_captioning(processor, model, device, image_path)
    print(f"Caption: {caption}\n")
    
    print("="*70)
    print("2. Visual Question Answering")
    print("="*70)
    
    question = "What colors are prominent in this image?"
    answer = visual_question_answering(processor, model, device, image_path, question)
    print(f"Q: {question}")
    print(f"A: {answer}\n")
    
    print("="*70)
    print("3. Detailed Scene Understanding")
    print("="*70)
    
    scene = scene_understanding(processor, model, device, image_path)
    print(f"Scene: {scene}\n")
    
    print("="*70)
    print("4. Multi-turn Conversation")
    print("="*70)
    
    chat_session = LLaVAChat(processor, model, device, image_path)
    
    # Turn 1
    response1 = chat_session.chat("What is the main subject of this image?")
    print(f"User: What is the main subject of this image?")
    print(f"Assistant: {response1}\n")
    
    # Turn 2 (이전 대화 기억)
    response2 = chat_session.chat("What color is it?")
    print(f"User: What color is it?")
    print(f"Assistant: {response2}\n")
    
    # Turn 3
    response3 = chat_session.chat("Where might this photo have been taken?")
    print(f"User: Where might this photo have been taken?")
    print(f"Assistant: {response3}\n")
    
    print("="*70)
    print("5. Creative/Diverse Response")
    print("="*70)
    
    creative = chat_with_options(
        processor, model, device, image_path,
        "Write a short creative story about this image.",
        temperature=0.8,  # 더 창의적
        do_sample=True
    )
    print(f"Story: {creative}\n")
    
    print("="*70)
    print("6. Object Counting")
    print("="*70)
    
    count = object_counting(processor, model, device, image_path, "people")
    print(f"Count: {count}\n")

if __name__ == "__main__":
    main()