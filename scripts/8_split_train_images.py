# scripts/8_split_train_images.py

import os
import shutil
import random
from tqdm import tqdm

# -------------------- 설정 --------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 원본 이미지 경로
SOURCE_DIR = os.path.join(PROJECT_ROOT, 'data', 'RoboFlow_FLIR_Dataset_v27_yolo11', 'train', 'images')

# 저장할 경로
DEST_DIR_1 = os.path.join(SOURCE_DIR, 'split_1')
DEST_DIR_2 = os.path.join(SOURCE_DIR, 'split_2')
DEST_DIR_3 = os.path.join(SOURCE_DIR, 'split_3')

# 랜덤 시드 고정
RANDOM_SEED = 42 
# ------------------------------------------------

def main():
    # 1. 폴더 생성
    for d in [DEST_DIR_1, DEST_DIR_2, DEST_DIR_3]:
        os.makedirs(d, exist_ok=True)

    # 2. 이미지 파일 리스트 로드
    all_files = sorted([f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    total_count = len(all_files)

    # 3. 랜덤 셔플 (시드 고정)
    random.seed(RANDOM_SEED)
    random.shuffle(all_files)

    # 4. 3등분으로 나누기
    split_size = total_count // 3
    
    group_1 = all_files[:split_size] # 7762장
    group_2 = all_files[split_size : split_size*2] # 7762장
    group_3 = all_files[split_size*2 :] # 7764장

    print(f"분할 정보:")
    print(f"   - Group 1 (split_1): {len(group_1)}장")
    print(f"   - Group 2 (split_2): {len(group_2)}장")
    print(f"   - Group 3 (split_3): {len(group_3)}장")

    # 5. 파일 복사
    # Group 1 복사
    for filename in tqdm(group_1, desc="Copying to split_1"):
        src = os.path.join(SOURCE_DIR, filename)
        dst = os.path.join(DEST_DIR_1, filename)
        shutil.copy2(src, dst)

    # Group 2 복사
    for filename in tqdm(group_2, desc="Copying to split_2"):
        src = os.path.join(SOURCE_DIR, filename)
        dst = os.path.join(DEST_DIR_2, filename)
        shutil.copy2(src, dst)

    # Group 3 복사
    for filename in tqdm(group_3, desc="Copying to split_3"):
        src = os.path.join(SOURCE_DIR, filename)
        dst = os.path.join(DEST_DIR_3, filename)
        shutil.copy2(src, dst)

    print("\n작업 완료")
    print(f"   확인 경로: {SOURCE_DIR} 내의 split_1, split_2, split_3 폴더")

if __name__ == '__main__':
    main()