# 8_random_inpainting.py

import os
import cv2
import numpy as np
import random
import argparse
from glob import glob
from tqdm import tqdm
from simple_lama_inpainting import SimpleLama
from PIL import Image

# ================= 기본 설정값 (Defaults) =================
DEFAULT_IN_IMG = '/home/vipmj/YOLO-Finetune/data/RoboFlow_FLIR_Dataset_v27_yolo11/train/images'
DEFAULT_IN_LBL = '/home/vipmj/YOLO-Finetune/data/RoboFlow_FLIR_Dataset_v27_yolo11/train/labels'
DEFAULT_OUT_IMG = '/home/vipmj/DATA/RoboFlow_Occlusion3/train/images'
DEFAULT_OUT_LBL = '/home/vipmj/DATA/RoboFlow_Occlusion3/train/labels'

# [중요] Occlusion을 적용할 대상의 Class ID 리스트
TARGET_CLASS_IDS = [0,1,2]  

lama = SimpleLama()
# ==========================================================

def get_args():
    parser = argparse.ArgumentParser(description="YOLO Dataset Random Occlusion Tool")
    
    # 1. 경로 설정 (Path Arguments)
    parser.add_argument('--input_images', type=str, default=DEFAULT_IN_IMG, help='입력 이미지 폴더 경로')
    parser.add_argument('--input_labels', type=str, default=DEFAULT_IN_LBL, help='입력 라벨 폴더 경로')
    parser.add_argument('--output_images', type=str, default=DEFAULT_OUT_IMG, help='출력 이미지 폴더 경로')
    parser.add_argument('--output_labels', type=str, default=DEFAULT_OUT_LBL, help='출력 라벨 폴더 경로')

    # 2. 파라미터 설정 (Hyperparameter Arguments)
    parser.add_argument('--prob', type=float, default=0.6, help='Occlusion 발생 확률 (0.0 ~ 1.0)')
    parser.add_argument('--min_ratio', type=float, default=0.4, help='Mask 최소 비율 (0.0 ~ 1.0)')
    parser.add_argument('--max_ratio', type=float, default=0.5, help='Mask 최대 비율 (0.0 ~ 1.0)')
    
    return parser.parse_args()

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def yolo_to_bbox(yolo_line, img_w, img_h):
    """YOLO 포맷(cls, x, y, w, h)을 픽셀 좌표(x1, y1, x2, y2)로 변환"""
    parts = list(map(float, yolo_line.strip().split()))
    cls_id = int(parts[0])
    x_c, y_c, w, h = parts[1], parts[2], parts[3], parts[4]
    
    w_px = w * img_w
    h_px = h * img_h
    x_c_px = x_c * img_w
    y_c_px = y_c * img_h
    
    x1 = int(x_c_px - w_px / 2)
    y1 = int(y_c_px - h_px / 2)
    x2 = int(x_c_px + w_px / 2)
    y2 = int(y_c_px + h_px / 2)
    
    return cls_id, max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)

def create_partial_mask(img_shape, bboxes, min_ratio, max_ratio):
    """선택된 객체들의 일부를 가리는 마스크 생성"""
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        
        if bw <= 0 or bh <= 0: continue

        ratio = random.uniform(min_ratio, max_ratio)
        mode = random.randint(0, 4)
        
        mx1, my1, mx2, my2 = x1, y1, x2, y2
        
        if mode == 0:   mx2 = x1 + int(bw * ratio) # 좌
        elif mode == 1: mx1 = x2 - int(bw * ratio) # 우
        elif mode == 2: my2 = y1 + int(bh * ratio) # 상
        elif mode == 3: my1 = y2 - int(bh * ratio) # 하
        else: # 내부
            mw, mh = int(bw * ratio), int(bh * ratio)
            mx1 = random.randint(x1, x2 - mw)
            my1 = random.randint(y1, y2 - mh)
            mx2, my2 = mx1 + mw, my1 + mh
            
        cv2.rectangle(mask, (mx1, my1), (mx2, my2), 255, -1)
        
    return mask

def process_dataset(args):
    # args에서 경로를 가져와 사용
    ensure_dir(args.output_images)
    ensure_dir(args.output_labels)
    
    # jpg, png, jpeg 모두 찾기
    img_paths = glob(os.path.join(args.input_images, '*.jpg')) + \
                glob(os.path.join(args.input_images, '*.png')) + \
                glob(os.path.join(args.input_images, '*.jpeg'))
    
    print(f"==================================================")
    print(f"Processing Images from: {args.input_images}")
    print(f"Saving Outputs to     : {args.output_images}")
    print(f"Found {len(img_paths)} images.")
    print(f"Params -> Prob: {args.prob}, Ratio: {args.min_ratio}~{args.max_ratio}")
    print(f"Target Classes: {TARGET_CLASS_IDS}")
    print(f"==================================================")
    
    for img_path in tqdm(img_paths):
        filename = os.path.basename(img_path)
        label_name = os.path.splitext(filename)[0] + '.txt'
        
        # 라벨 경로는 args.input_labels 사용
        label_path = os.path.join(args.input_labels, label_name)
        
        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        targets_px = []
        original_lines = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    original_lines.append(line)
                    
                    cls_id, x1, y1, x2, y2 = yolo_to_bbox(line, w, h)
                    
                    if (cls_id in TARGET_CLASS_IDS) and (random.random() < args.prob):
                        targets_px.append((x1, y1, x2, y2))
        
        # 가릴 대상이 없으면 원본 저장
        if not targets_px:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(args.output_images, filename), img_bgr)
            with open(os.path.join(args.output_labels, label_name), 'w') as f:
                f.writelines(original_lines)
            continue

        # 1. 마스크 생성
        mask = create_partial_mask(img.shape, targets_px, args.min_ratio, args.max_ratio)
        
        # 2. LaMa Inpainting
        img_pil = Image.fromarray(img)
        mask_pil = Image.fromarray(mask)
        
        result_pil = lama(img_pil, mask_pil)
        result_np = np.array(result_pil)
        
        # 3. 저장
        result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.output_images, filename), result_bgr)
        
        # 라벨 저장
        with open(os.path.join(args.output_labels, label_name), 'w') as f:
            f.writelines(original_lines)

if __name__ == "__main__":
    args = get_args()
    process_dataset(args)