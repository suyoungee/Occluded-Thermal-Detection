import os
import glob

# ---------------- Configuration ----------------
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'RoboFlow_FLIR_Dataset_v27_yolo11')

IMAGES_ROOT = os.path.join(DATA_ROOT, 'images')
LABELS_ROOT = os.path.join(DATA_ROOT, 'labels')

SPLITS = ['train', 'val', 'test']

def check_labels_integrity():
    print("🧐 Labels <-> Images 정합성 및 중복 검사를 시작합니다...\n")
    
    label_sets = {}
    
    # 1. Check image-label pairing for each split.
    for split in SPLITS:
        print(f"--- [ {split.upper()} Set 검사 ] ---")
        
        img_dir = os.path.join(IMAGES_ROOT, split)
        lbl_dir = os.path.join(LABELS_ROOT, split)
        
        # Collect filenames without extensions.
        # Extend this pattern when image formats other than JPG are used.
        img_files = {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(img_dir, '*.jpg'))}
        lbl_files = {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(lbl_dir, '*.txt'))}
        
        label_sets[split] = lbl_files # Retain labels for the overlap check.

        # A. Labels without images are fatal during training.
        orphaned_labels = lbl_files - img_files
        if orphaned_labels:
            print(f"❌ [심각] 이미지가 없는 라벨(Orphaned Labels) 발견: {len(orphaned_labels)}개")
            # Print at most three examples.
            print(f"   예시: {list(orphaned_labels)[:3]} ...")
        else:
            print("✅ 모든 라벨 파일에 대응하는 이미지가 존재합니다.")

        # B. YOLO treats images without labels as background images.
        # Because 1_convert_data.py copies only images containing objects, a
        # mismatch here may indicate a conversion or file-copying issue.
        orphaned_images = img_files - lbl_files
        if orphaned_images:
            print(f"⚠️ [주의] 라벨이 없는 이미지(Background Images?) 발견: {len(orphaned_images)}개")
        else:
            print("✅ 모든 이미지 파일에 대응하는 라벨이 존재합니다.")
            
        print(f"   -> 라벨 파일 수: {len(lbl_files):,}개 / 이미지 파일 수: {len(img_files):,}개")
        print("")

    # 2. Check label overlap between splits.
    print("--- [ Labels 중복(Data Leakage) 검사 ] ---")
    overlap_found = False
    
    # Train vs Val
    train_val = label_sets['train'].intersection(label_sets['val'])
    if train_val:
        print(f"❌ [심각] Train과 Val 라벨 간 중복 발생: {len(train_val)}개")
        overlap_found = True
    else:
        print("✅ Train ∩ Val: 라벨 중복 없음")

    # Train vs Test
    train_test = label_sets['train'].intersection(label_sets['test'])
    if train_test:
        print(f"❌ [심각] Train과 Test 라벨 간 중복 발생: {len(train_test)}개")
        overlap_found = True
    else:
        print("✅ Train ∩ Test: 라벨 중복 없음")

    # Val vs Test
    val_test = label_sets['val'].intersection(label_sets['test'])
    if val_test:
        print(f"❌ [심각] Val과 Test 라벨 간 중복 발생: {len(val_test)}개")
        overlap_found = True
    else:
        print("✅ Val ∩ Test: 라벨 중복 없음")
        
    if not overlap_found:
        print("\n✨ 라벨 데이터셋이 완벽하게 분리되었습니다.")
    else:
        print("\n🚨 중복된 파일이 존재합니다. 데이터 분할 로직을 다시 확인하세요.")

if __name__ == '__main__':
    if os.path.exists(DATA_ROOT):
        check_labels_integrity()
    else:
        print(f"🚨 데이터 경로를 찾을 수 없습니다: {DATA_ROOT}")
