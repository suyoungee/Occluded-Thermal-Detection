import pandas as pd
import os
import glob
import shutil
from PIL import Image
import random
from tqdm import tqdm  # 진행률 표시를 위해 추가 (없으면 pip install tqdm)

# ----------------- 1. 설정 및 경로 정의 -----------------
# 프로젝트의 타겟 데이터 경로 (프로젝트 루트 기준)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TARGET_DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'RoboFlow_FLIR_Dataset_v27_yolo11')

# 원본 데이터셋 루트 경로 (사용자 제공 경로)
ORIGINAL_ROOT = os.path.join(PROJECT_ROOT, 'data', 'RoboFlow_FLIR_Dataset_v27_yolo11')

# YOLO 타겟 폴더 구조
TARGET_IMAGES = os.path.join(TARGET_DATA_ROOT, 'images')
TARGET_LABELS = os.path.join(TARGET_DATA_ROOT, 'labels')

# CSV 컬럼 정의
CSV_COLUMNS = [
    'frame_number', 'object_id', 'x', 'y', 'w', 'h', 
    'class_id', 'species', 'occlusion', 'noise'
]

# Real 데이터셋 분할 설정
REAL_VAL_COUNT = 10  # 10개를 Val로, 나머지를 Train으로

# ----------------- 2. 유틸리티 함수 -----------------

def mot_to_yolo(x, y, w, h, W, H):
    """MOT BBox (x, y, w, h) -> YOLO 정규화 (x_c, y_c, w_n, h_n)"""
    if W == 0 or H == 0: return None
    x_center = (x + w / 2) / W
    y_center = (y + h / 2) / H
    width = w / W
    height = h / H
    return f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def setup_directories():
    """타겟 디렉토리 초기화"""
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(TARGET_IMAGES, split), exist_ok=True)
        os.makedirs(os.path.join(TARGET_LABELS, split), exist_ok=True)

# ----------------- 3. 메인 처리 로직 -----------------

def process_subset(subset_folder, split_name, allowed_videos=None):
    """
    특정 서브셋 폴더(TrainReal 등)를 처리하여 Target Split(train/val/test)으로 변환
    """
    print(f"\n🚀 처리 시작: {subset_folder} -> {split_name}")
    
    # 어노테이션 폴더 경로
    anno_dir = os.path.join(ORIGINAL_ROOT, subset_folder, 'annotations')
    img_root_dir = os.path.join(ORIGINAL_ROOT, subset_folder, 'images')
    
    # CSV 파일 리스트 가져오기
    csv_files = glob.glob(os.path.join(anno_dir, '*.csv'))
    
    # 진행률 표시바 생성
    pbar = tqdm(csv_files, desc=f"{split_name} processing")
    
    file_count = 0
    
    for csv_path in pbar:
        # CSV 파일명 = Video ID (예: 0000000067_0000000002.csv -> 0000000067_0000000002)
        video_id = os.path.basename(csv_path).replace('.csv', '')
        
        # 비디오 필터링 (TrainReal 분할용)
        if allowed_videos is not None and video_id not in allowed_videos:
            continue
            
        # 이미지 폴더 경로 확인 (예: /TrainReal/images/0000000067_0000000002/)
        video_img_dir = os.path.join(img_root_dir, video_id)
        if not os.path.isdir(video_img_dir):
            # print(f"  [Skip] 이미지 폴더 없음: {video_id}")
            continue

        # CSV 로드
        try:
            df = pd.read_csv(csv_path, header=None, names=CSV_COLUMNS)
        except Exception as e:
            print(f"  [Error] CSV 읽기 실패 {csv_path}: {e}")
            continue

        # -------------------------------------------------------
        # 중요: 이미지 파일별로 어노테이션을 모으기 위한 딕셔너리
        # (한 이미지에 여러 객체가 있을 수 있으므로)
        # Key: 이미지 파일명, Value: list of YOLO strings
        # -------------------------------------------------------
        img_labels_map = {}

        for _, row in df.iterrows():
            species = int(row['species'])
            
            # species 필터링 (-1: Unknown 제외, 0~8만 사용)
            if species < 0 or species > 8:
                continue

            # 프레임 번호 (예: 87)
            frame_num = int(row['frame_number'])
            
            # -------------------------------------------------------
            # [핵심] 파일명 생성 (10자리 패딩)
            # CSV: 87 -> Filename: VideoID_0000000087.jpg
            # -------------------------------------------------------
            img_filename = f"{video_id}_{frame_num:010d}.jpg"
            
            # 원본 이미지 전체 경로
            src_img_path = os.path.join(video_img_dir, img_filename)
            
            # 파일 존재 여부 확인 (Sim 데이터의 rgb/seg/png 등은 여기서 자연스럽게 걸러짐)
            if not os.path.exists(src_img_path):
                continue
                
            # 이미지 크기 확인 (최초 1회만 수행하기 위해 캐싱하면 좋지만, 단순화를 위해 매번 체크하거나 PIL 사용)
            # 주의: Sim 데이터는 이미지가 다를 수 있으므로 실제 파일을 열어봐야 함
            try:
                with Image.open(src_img_path) as img:
                    W, H = img.size
            except:
                continue
            
            # 좌표 변환
            yolo_bbox = mot_to_yolo(row['x'], row['y'], row['w'], row['h'], W, H)
            if yolo_bbox is None: continue
            
            # 리스트에 추가
            if img_filename not in img_labels_map:
                img_labels_map[img_filename] = []
            
            # YOLO Format: class_id center_x center_y w h
            img_labels_map[img_filename].append(f"{species} {yolo_bbox}\n")

        # -------------------------------------------------------
        # 파일 복사 및 .txt 생성 (해당 비디오의 모든 프레임 처리)
        # -------------------------------------------------------
        for img_filename, label_lines in img_labels_map.items():
            # 1. 이미지 복사
            src_img = os.path.join(video_img_dir, img_filename)
            dst_img = os.path.join(TARGET_IMAGES, split_name, img_filename)
            
            if not os.path.exists(dst_img):
                shutil.copy2(src_img, dst_img)
            
            # 2. 라벨 파일 생성 (.txt)
            txt_filename = img_filename.replace('.jpg', '.txt')
            dst_label = os.path.join(TARGET_LABELS, split_name, txt_filename)
            
            with open(dst_label, 'w') as f:
                f.writelines(label_lines)
                
        file_count += 1
        
    print(f"✅ {split_name} 완료: {file_count}개의 비디오 처리됨.")

def main():
    # 1. 디렉토리 생성
    setup_directories()
    
    # 2. TrainReal 비디오 리스트 가져오기 및 분할 (Train/Val)
    train_real_anno_path = os.path.join(ORIGINAL_ROOT, 'TrainReal', 'annotations')
    
    if not os.path.exists(train_real_anno_path):
        print(f"🚨 오류: 경로를 찾을 수 없습니다: {train_real_anno_path}")
        return

    # 전체 CSV 목록 (경로 제외한 파일명만, 확장자 제거)
    all_real_videos = [os.path.basename(x).replace('.csv', '') for x in glob.glob(os.path.join(train_real_anno_path, '*.csv'))]
    
    # 랜덤 셔플 및 분할
    random.seed(42)
    random.shuffle(all_real_videos)
    
    # 검증셋 10개, 나머지 학습셋
    val_videos = all_real_videos[:REAL_VAL_COUNT]
    train_videos = all_real_videos[REAL_VAL_COUNT:]
    
    print(f"📋 데이터 분할 정보:")
    print(f"   - Total Real Videos: {len(all_real_videos)}")
    print(f"   - Train Real: {len(train_videos)}")
    print(f"   - Val Real: {len(val_videos)}")
    
    # 3. 변환 실행
    
    # (1) TrainReal -> train (38개)
    process_subset('TrainReal', 'train', allowed_videos=train_videos)
    
    # (2) TrainReal -> val (10개)
    process_subset('TrainReal', 'val', allowed_videos=val_videos)
    
    # (3) TrainSimulation -> train (전체)
    # rgb, seg, png 파일은 process_subset 내부 로직에 의해 자동 제외됨 (.jpg만 타겟팅)
    process_subset('TrainSimulation', 'train', allowed_videos=None)
    
    # (4) TestReal -> test (전체)
    process_subset('TestReal', 'test', allowed_videos=None)

    print("\n🎉 모든 데이터 변환이 완료되었습니다.")

if __name__ == '__main__':
    main()