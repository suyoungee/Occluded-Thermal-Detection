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

# ---------------- Configuration ----------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Input shard created by the image-splitting script
# (/train/images/split_1, split_2, or split_3).
SPLIT_FOLDER = 'split_1'
INPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'RoboFlow_FLIR_Dataset_v27_yolo11', 'train', 'images', SPLIT_FOLDER)

# Output directory for generated images.
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'FLUX1-Fill-dev', SPLIT_FOLDER)

NUM_IMAGES_TO_GENERATE = 800 # Generate 800 images from each shard.

MODEL_ID = "black-forest-labs/FLUX.1-Fill-dev"
# -----------------------------------------------------

def draw_fractal_branch(img, start_pt, angle, length, thickness, depth):
    # Recursively branch the lines to create a fractal tree.
    if depth == 0 or length < 2:
        return

    # Calculate the branch endpoint.
    end_x = int(start_pt[0] + length * math.cos(math.radians(angle)))
    end_y = int(start_pt[1] + length * math.sin(math.radians(angle)))
    end_pt = (end_x, end_y)

    # Draw branches that become thinner toward their tips.
    current_thickness = max(1, int(thickness))
    cv2.line(img, start_pt, end_pt, 255, current_thickness)

    # Shorten and narrow child branches.
    new_length = length * random.uniform(0.65, 0.85)
    new_thickness = thickness * 0.7
    
    # Split each branch two or three ways for an irregular thicket pattern.
    num_splits = random.randint(2, 3)
    
    for _ in range(num_splits):
        # Spread child branches by 15 to 45 degrees.
        angle_change = random.randint(15, 45) * random.choice([-1, 1])
        # Add a small random angle offset.
        new_angle = angle + angle_change + random.randint(-5, 5)
        
        draw_fractal_branch(img, end_pt, new_angle, new_length, new_thickness, depth - 1)

def create_fractal_mask(h, w):
    # Create a mask of thin, tangled foreground branches.
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Grow four or five trees inward from the image edges.
    num_trees = random.randint(4, 5)
    
    for _ in range(num_trees):
        # Choose a starting point along the top, bottom, left, or right edge.
        side = random.choice(['top', 'bottom', 'left', 'right'])
        
        if side == 'top':
            start_pt = (random.randint(0, w), 0)
            base_angle = 90 # Downward.
        elif side == 'bottom':
            start_pt = (random.randint(0, w), h)
            base_angle = -90 # Upward.
        elif side == 'left':
            start_pt = (0, random.randint(0, h))
            base_angle = 0 # To the right.
        else: # Right.
            start_pt = (w, random.randint(0, h))
            base_angle = 180 # To the left.

        draw_fractal_branch(
            mask, 
            start_pt, 
            base_angle + random.randint(-20, 20), 
            length=random.randint(100, 180), 
            thickness=random.randint(10, 20), 
            depth=5 # Larger values create more branches.
        )
            
    # Slightly blur the mask to produce a more natural FLUX transition.
    return Image.fromarray(mask).filter(ImageFilter.GaussianBlur(radius=1))

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(INPUT_DIR):
        print(f"오류: 입력 경로 없음")
        return

    all_images = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    selected_images = random.sample(all_images, NUM_IMAGES_TO_GENERATE) # Select a random subset.

    print(f"FLUX.1-Fill-dev [Dual-GPU Balanced Mode] 로딩 중...")

    try:
        # 1. Configure 4-bit quantization.
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        # 2. Load and distribute the text encoder.
        text_encoder_2 = T5EncoderModel.from_pretrained(
            MODEL_ID,
            subfolder="text_encoder_2",
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            device_map="balanced"
        )

        # 3. Load and distribute the transformer.
        transformer = FluxTransformer2DModel.from_pretrained(
            MODEL_ID,
            subfolder="transformer",
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            device_map="balanced"
        )

        # 4. Assemble the fill pipeline.
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

    # Keep generation at 768 pixels to limit memory usage.
    GEN_SIZE = 768 

    for img_file in tqdm(selected_images):
        img_path = os.path.join(INPUT_DIR, img_file)
        
        try:
            # 1. Release cached memory.
            gc.collect()
            torch.cuda.empty_cache()

            original_image = Image.open(img_path).convert("RGB")
            original_size = original_image.size 
            
            # 2. Resize to the generation resolution.
            target_size = (GEN_SIZE, GEN_SIZE)
            init_image_resized = original_image.resize(target_size)
            
            # 3. Create a fractal branch mask.
            mask_image = create_fractal_mask(GEN_SIZE, GEN_SIZE)
            
            # 4. Describe tangled branches, fog, and thermal sensor texture.
            prompt = (
                "thermal imaging texture, grayscale. "
                "View through a thicket of thin, dry, withered tree twigs and tangled branches in the foreground, "
                "covered in dense fog and thick mist. "
                "The atmosphere is hazy, with low visibility, obscuring the background. "
                "Atmospheric effect, heavy grain, noisy thermal sensor style." 
            )

            # 5. Generate the masked content.
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
            
            # 6. Composite and save the result.
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
