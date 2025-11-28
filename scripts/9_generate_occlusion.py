import os
import cv2
import torch
import random
import numpy as np
import gc
import math
from PIL import Image, ImageFilter
from tqdm import tqdm

from diffusers import FluxFillPipeline, FluxTransformer2DModel
from transformers import T5EncoderModel, BitsAndBytesConfig

# -------------------- 설정 --------------------
# 입력 이미지 경로 (/train/images/split_1,split_2, split_3) <- 8번 코드로 분할한 이미지 폴더 두 개
# 수영: split_1 담당, 민중: split_2 담당 , 학연생 컴퓨터: split_3 담당
SPLIT_FOLDER = 'split_1'
INPUT_DIR = os.path.join('/home/suyoung/study/IPIU-2026/data/RoboFlow_FLIR_Dataset_v27_yolo11/train/images', SPLIT_FOLDER)

# 생성한 이미지 저장 경로
OUTPUT_DIR = os.path.join('/home/suyoung/study/IPIU-2026/data/FLUX1-Fill-dev', SPLIT_FOLDER)

# NUM_IMAGES_TO_GENERATE = 2400
NUM_IMAGES_TO_GENERATE = 800 # 2400장에서 800장으로 변경

MODEL_ID = "black-forest-labs/FLUX.1-Fill-dev"
# -----------------------------------------------------

def draw_fractal_branch(img, start_pt, angle, length, thickness, depth):
    # 재귀 함수를 이용해 진짜 나무처럼 가지 치기 (Fractal Tree)
    if depth == 0 or length < 2:
        return

    # 끝점 계산
    end_x = int(start_pt[0] + length * math.cos(math.radians(angle)))
    end_y = int(start_pt[1] + length * math.sin(math.radians(angle)))
    end_pt = (end_x, end_y)

    # 선 그리기 (끝으로 갈수록 얇아짐)
    current_thickness = max(1, int(thickness))
    cv2.line(img, start_pt, end_pt, 255, current_thickness)

    # 가지 치기 설정
    # 길이는 점점 짧게, 두께는 점점 얇게
    new_length = length * random.uniform(0.65, 0.85)
    new_thickness = thickness * 0.7
    
    # 가지가 2~3갈래로 나뉨 (자연스러운 수풀 느낌)
    num_splits = random.randint(2, 3)
    
    for _ in range(num_splits):
        # 각도 변화 (15~40도 사이로 벌어짐)
        angle_change = random.randint(15, 45) * random.choice([-1, 1])
        # 약간의 랜덤성 추가
        new_angle = angle + angle_change + random.randint(-5, 5)
        
        draw_fractal_branch(img, end_pt, new_angle, new_length, new_thickness, depth - 1)

def create_fractal_mask(h, w):
    # 앞을 가로막는 얇고 복잡한 잔가지' 마스크 생성
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 화면 밖에서 안으로 뻗어 들어오는 나무 4-5 그루 생성
    num_trees = random.randint(4, 5)
    
    for _ in range(num_trees):
        # 시작점: 화면 가장자리 (위, 아래, 좌, 우)
        side = random.choice(['top', 'bottom', 'left', 'right'])
        
        if side == 'top':
            start_pt = (random.randint(0, w), 0)
            base_angle = 90 # 아래로
        elif side == 'bottom':
            start_pt = (random.randint(0, w), h)
            base_angle = -90 # 위로
        elif side == 'left':
            start_pt = (0, random.randint(0, h))
            base_angle = 0 # 오른쪽으로
        else: # right
            start_pt = (w, random.randint(0, h))
            base_angle = 180 # 왼쪽으로

        draw_fractal_branch(
            mask, 
            start_pt, 
            base_angle + random.randint(-20, 20), 
            length=random.randint(100, 180), 
            thickness=random.randint(10, 20), 
            depth=5 # 값 클수록 더 많은 잔가지
        )
            
    # 살짝 블러를 줘서 FLUX가 자연스럽게 인식하도록 함
    return Image.fromarray(mask).filter(ImageFilter.GaussianBlur(radius=1))

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(INPUT_DIR):
        print(f"오류: 입력 경로 없음")
        return

    all_images = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    selected_images = random.sample(all_images, NUM_IMAGES_TO_GENERATE) # 랜덤으로 n장 선택

    print(f"FLUX.1-Fill-dev [Dual-GPU Balanced Mode] 로딩 중...")

    try:
        # 1. 4-bit 양자화 설정
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        # 2. Text Encoder 분산 로드
        text_encoder_2 = T5EncoderModel.from_pretrained(
            MODEL_ID,
            subfolder="text_encoder_2",
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            device_map="balanced"
        )

        # 3. Transformer 분산 로드
        transformer = FluxTransformer2DModel.from_pretrained(
            MODEL_ID,
            subfolder="transformer",
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            device_map="balanced"
        )

        # 4. 파이프라인 조립
        pipe = FluxFillPipeline.from_pretrained(
            MODEL_ID,
            transformer=transformer,
            text_encoder_2=text_encoder_2,
            torch_dtype=torch.float16,
            device_map="balanced"
        )
        
    except Exception as e:
        print(f"모델 로딩 실패: {e}")
        return

    print(f"{len(selected_images)}장 생성 시작...")

    # 메모리 확보를 위해 768 유지
    GEN_SIZE = 768 

    for img_file in tqdm(selected_images):
        img_path = os.path.join(INPUT_DIR, img_file)
        
        try:
            # 1. 메모리 청소
            gc.collect()
            torch.cuda.empty_cache()

            original_image = Image.open(img_path).convert("RGB")
            original_size = original_image.size 
            
            # 2. 생성 해상도 조정
            target_size = (GEN_SIZE, GEN_SIZE)
            init_image_resized = original_image.resize(target_size)
            
            # 3. 프랙탈 마스크 생성
            mask_image = create_fractal_mask(GEN_SIZE, GEN_SIZE)
            
            # 4. 프롬프트 튜닝 (안개보단 나뭇가지의 복잡함에 집중. 얇고 불규칙한 선들이 객체의 윤곽선을 끊어먹도록 유도)
            prompt = (
                "thermal imaging texture, grayscale. "
                "View through a thicket of thin, dry, withered tree twigs and tangled branches in the foreground, "
                "covered in dense fog and thick mist. "
                "The atmosphere is hazy, with low visibility, obscuring the background. "
                "Atmospheric effect, heavy grain, noisy thermal sensor style." 
            )

            """
            나뭇가지만 있는 프롬프트
            prompt = (
                "thermal imaging texture, grayscale. "
                "View through a thicket of thin, dry, withered tree twigs and tangled branches in the foreground. "
                "The branches are thin, intricate, overlapping, and blocking the view. "
                "Depth of field, blurry foreground branches, heavy grain, noisy thermal sensor style."
            )
            """
            
            # 5. 생성
            generated_image = pipe(
                prompt=prompt,
                image=init_image_resized,
                mask_image=mask_image,
                height=GEN_SIZE,
                width=GEN_SIZE,
                guidance_scale=30.0, 
                num_inference_steps=40,
                max_sequence_length=256
            ).images[0]
            
            # 6. 합성 및 저장
            generated_image = generated_image.resize(original_size)
            generated_image = generated_image.convert("L")
            mask_resized = mask_image.resize(original_size)
            final_image = Image.composite(generated_image.convert("RGB"), original_image, mask_resized)
            final_image = final_image.convert("L") 
            
            save_name = f"{img_file}"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            final_image.save(save_path)
            
        except Exception as e:
            print(f"{img_file} 에러: {e}")
            gc.collect()
            torch.cuda.empty_cache()
            continue

    print("\n작업 완료")
    print(f"저장 위치: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()