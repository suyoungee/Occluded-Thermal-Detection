import os
import glob

# ----------------- 설정 -----------------
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'RoboFlow_FLIR_Dataset_v27_yolo11')

IMAGES_ROOT = os.path.join(DATA_ROOT, 'images')
LABELS_ROOT = os.path.join(DATA_ROOT, 'labels')

SPLITS = ['train', 'val', 'test']

def check_labels_integrity():
    print("🧐 Labels <-> Images 정합성 및 중복 검사를 시작합니다...\n")
    
    label_sets = {}
    
    # 1. 각 Split별 정합성(Pairing) 검사
    for split in SPLITS:
        print(f"--- [ {split.upper()} Set 검사 ] ---")
        
        img_dir = os.path.join(IMAGES_ROOT, split)
        lbl_dir = os.path.join(LABELS_ROOT, split)
        
        # 파일 목록 가져오기 (확장자 제거한 순수 파일명)
        # jpg 외에 다른 이미지 포맷이 있다면 확장자 리스트 조정 필요
        img_files = {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(img_dir, '*.jpg'))}
        lbl_files = {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(lbl_dir, '*.txt'))}
        
        label_sets[split] = lbl_files # 중복 검사를 위해 저장
        
        # A. Labels without Images (치명적 오류)
        # 라벨은 있는데 이미지가 없는 경우 -> 학습 시 에러 발생
        orphaned_labels = lbl_files - img_files
        if orphaned_labels:
            print(f"❌ [심각] 이미지가 없는 라벨(Orphaned Labels) 발견: {len(orphaned_labels)}개")
            # 예시 3개만 출력
            print(f"   예시: {list(orphaned_labels)[:3]} ...")
        else:
            print("✅ 모든 라벨 파일에 대응하는 이미지가 존재합니다.")

        # B. Images without Labels (경고/참고)
        # 이미지는 있는데 라벨이 없는 경우 -> YOLO는 이를 '배경(Background)' 이미지로 간주함.
        # 하지만 1_convert_data.py 로직상 객체가 있는 이미지만 복사했으므로, 
        # 여기서 차이가 나면 변환 과정이나 파일 복사 과정의 문제일 수 있음.
        orphaned_images = img_files - lbl_files
        if orphaned_images:
            print(f"⚠️ [주의] 라벨이 없는 이미지(Background Images?) 발견: {len(orphaned_images)}개")
            # print(f"   (의도한 배경 이미지가 아니라면 1_convert_data.py를 확인하세요.)")
        else:
            print("✅ 모든 이미지 파일에 대응하는 라벨이 존재합니다.")
            
        print(f"   -> 라벨 파일 수: {len(lbl_files):,}개 / 이미지 파일 수: {len(img_files):,}개")
        print("")

    # 2. Labels 간 중복(Leakage) 검사
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