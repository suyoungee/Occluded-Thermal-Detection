import pandas as pd
import os
import glob
import shutil
from PIL import Image
import random
from tqdm import tqdm  # Displays progress bars (install with pip if missing).

# ----------------- 1. Configuration and paths -----------------
# Target dataset path relative to the project root.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TARGET_DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'RoboFlow_FLIR_Dataset_v27_yolo11')

# Root directory of the source dataset.
# By default this matches TARGET_DATA_ROOT; change it when converting from a different raw dataset layout.
ORIGINAL_ROOT = os.path.join(PROJECT_ROOT, 'data', 'RoboFlow_FLIR_Dataset_v27_yolo11')
# YOLO destination directories.
TARGET_IMAGES = os.path.join(TARGET_DATA_ROOT, 'images')
TARGET_LABELS = os.path.join(TARGET_DATA_ROOT, 'labels')

# CSV column definitions.
CSV_COLUMNS = [
    'frame_number', 'object_id', 'x', 'y', 'w', 'h', 
    'class_id', 'species', 'occlusion', 'noise'
]

# Real-dataset split configuration.
REAL_VAL_COUNT = 10  # Use 10 videos for validation and the rest for training.

# ----------------- 2. Utility functions -----------------

def mot_to_yolo(x, y, w, h, W, H):
    """Convert a MOT box (x, y, w, h) to normalized YOLO coordinates."""
    if W == 0 or H == 0: return None
    x_center = (x + w / 2) / W
    y_center = (y + h / 2) / H
    width = w / W
    height = h / H
    return f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def setup_directories():
    """Create the destination directories."""
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(TARGET_IMAGES, split), exist_ok=True)
        os.makedirs(os.path.join(TARGET_LABELS, split), exist_ok=True)

# ----------------- 3. Main processing logic -----------------

def process_subset(subset_folder, split_name, allowed_videos=None):
    """
    Convert a source subset such as TrainReal to a target train, val, or test
    split.
    """
    print(f"\n🚀 처리 시작: {subset_folder} -> {split_name}")
    
    # Source annotation and image directories.
    anno_dir = os.path.join(ORIGINAL_ROOT, subset_folder, 'annotations')
    img_root_dir = os.path.join(ORIGINAL_ROOT, subset_folder, 'images')
    
    # Collect annotation CSV files.
    csv_files = glob.glob(os.path.join(anno_dir, '*.csv'))
    
    # Create a progress bar.
    pbar = tqdm(csv_files, desc=f"{split_name} processing")
    
    file_count = 0
    
    for csv_path in pbar:
        # The CSV stem is the video ID.
        video_id = os.path.basename(csv_path).replace('.csv', '')
        
        # Filter videos when splitting TrainReal.
        if allowed_videos is not None and video_id not in allowed_videos:
            continue
            
        # Check the corresponding video image directory.
        video_img_dir = os.path.join(img_root_dir, video_id)
        if not os.path.isdir(video_img_dir):
            continue

        # Load annotations.
        try:
            df = pd.read_csv(csv_path, header=None, names=CSV_COLUMNS)
        except Exception as e:
            print(f"  [Error] CSV 읽기 실패 {csv_path}: {e}")
            continue

        # Group annotations by image because one image may contain multiple
        # objects. Keys are image filenames and values are YOLO label lines.
        img_labels_map = {}

        for _, row in df.iterrows():
            species = int(row['species'])
            
            # Exclude unknown species and retain IDs from 0 through 8.
            if species < 0 or species > 8:
                continue

            # Frame number, for example 87.
            frame_num = int(row['frame_number'])
            
            # Pad frame numbers to 10 digits when constructing filenames.
            # Example: 87 -> VideoID_0000000087.jpg.
            img_filename = f"{video_id}_{frame_num:010d}.jpg"
            
            # Full source image path.
            src_img_path = os.path.join(video_img_dir, img_filename)
            
            # Missing files and non-target simulation assets are skipped here.
            if not os.path.exists(src_img_path):
                continue
                
            # Open each file because simulation images may have different sizes.
            try:
                with Image.open(src_img_path) as img:
                    W, H = img.size
            except:
                continue
            
            # Convert box coordinates.
            yolo_bbox = mot_to_yolo(row['x'], row['y'], row['w'], row['h'], W, H)
            if yolo_bbox is None: continue
            
            # Append the label to this image's annotation list.
            if img_filename not in img_labels_map:
                img_labels_map[img_filename] = []
            
            # YOLO Format: class_id center_x center_y w h
            img_labels_map[img_filename].append(f"{species} {yolo_bbox}\n")

        # Copy images and write a label file for every annotated frame.
        for img_filename, label_lines in img_labels_map.items():
            # 1. Copy the image.
            src_img = os.path.join(video_img_dir, img_filename)
            dst_img = os.path.join(TARGET_IMAGES, split_name, img_filename)
            
            if not os.path.exists(dst_img):
                shutil.copy2(src_img, dst_img)
            
            # 2. Write the label file.
            txt_filename = img_filename.replace('.jpg', '.txt')
            dst_label = os.path.join(TARGET_LABELS, split_name, txt_filename)
            
            with open(dst_label, 'w') as f:
                f.writelines(label_lines)
                
        file_count += 1
        
    print(f"✅ {split_name} 완료: {file_count}개의 비디오 처리됨.")

def main():
    # 1. Create destination directories.
    setup_directories()
    
    # 2. Collect and split TrainReal videos into train and validation sets.
    train_real_anno_path = os.path.join(ORIGINAL_ROOT, 'TrainReal', 'annotations')
    
    if not os.path.exists(train_real_anno_path):
        print(f"🚨 오류: 경로를 찾을 수 없습니다: {train_real_anno_path}")
        return

    # Collect CSV stems without directory paths or extensions.
    all_real_videos = [os.path.basename(x).replace('.csv', '') for x in glob.glob(os.path.join(train_real_anno_path, '*.csv'))]
    
    # Shuffle deterministically before splitting.
    random.seed(42)
    random.shuffle(all_real_videos)
    
    # Use 10 videos for validation and the rest for training.
    val_videos = all_real_videos[:REAL_VAL_COUNT]
    train_videos = all_real_videos[REAL_VAL_COUNT:]
    
    print(f"📋 데이터 분할 정보:")
    print(f"   - Total Real Videos: {len(all_real_videos)}")
    print(f"   - Train Real: {len(train_videos)}")
    print(f"   - Val Real: {len(val_videos)}")
    
    # 3. Run conversion.
    
    # (1) TrainReal -> train (38 videos).
    process_subset('TrainReal', 'train', allowed_videos=train_videos)
    
    # (2) TrainReal -> val (10 videos).
    process_subset('TrainReal', 'val', allowed_videos=val_videos)
    
    # (3) TrainSimulation -> train (all videos).
    # RGB, segmentation, and PNG assets are skipped because only matching JPG
    # files are processed.
    process_subset('TrainSimulation', 'train', allowed_videos=None)
    
    # (4) TestReal -> test (all videos).
    process_subset('TestReal', 'test', allowed_videos=None)

    print("\n🎉 모든 데이터 변환이 완료되었습니다.")

if __name__ == '__main__':
    main()
