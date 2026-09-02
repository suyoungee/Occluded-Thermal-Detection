import os
import glob

# ---------------- Configuration ----------------
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'RoboFlow_FLIR_Dataset_v27_yolo11', 'images')

# Approximate frame count reported in the reference paper.
# Real: ~62k, Sim: ~100k -> Total: ~162k
# The exact count may vary slightly between dataset versions.
EXPECTED_TOTAL_FRAMES_APPROX = 162000 

def check_integrity():
    print("🧐 데이터셋 무결성 검사를 시작합니다...\n")
    
    splits = ['train', 'val', 'test']
    file_sets = {}
    total_files = 0
    
    # 1. Collect files from each split.
    for split in splits:
        split_path = os.path.join(DATA_ROOT, split)
        
        if not os.path.exists(split_path):
            print(f"🚨 경고: {split} 폴더가 존재하지 않습니다! ({split_path})")
            file_sets[split] = set()
            continue
            
        # Collect image files.
        files = glob.glob(os.path.join(split_path, '*.jpg'))
        # Keep filenames without their directory paths.
        filenames = [os.path.basename(f) for f in files]
        
        file_sets[split] = set(filenames)
        count = len(filenames)
        total_files += count
        
        print(f"📁 {split.upper()} 셋 파일 수: {count:,} 개")

    print(f"\n📊 총 이미지 파일 수: {total_files:,} 개")
    
    # 2. Check mutual exclusivity.
    print("\n🔍 중복 파일 검사 중...")
    overlap_found = False
    
    # Train vs Val
    train_val_overlap = file_sets['train'].intersection(file_sets['val'])
    if train_val_overlap:
        print(f"❌ [심각] Train과 Val 사이에 {len(train_val_overlap)}개의 중복 파일이 있습니다!")
        overlap_found = True
    else:
        print("✅ Train ∩ Val: 중복 없음")
        
    # Train vs Test
    train_test_overlap = file_sets['train'].intersection(file_sets['test'])
    if train_test_overlap:
        print(f"❌ [심각] Train과 Test 사이에 {len(train_test_overlap)}개의 중복 파일이 있습니다!")
        overlap_found = True
    else:
        print("✅ Train ∩ Test: 중복 없음")
        
    # Val vs Test
    val_test_overlap = file_sets['val'].intersection(file_sets['test'])
    if val_test_overlap:
        print(f"❌ [심각] Val과 Test 사이에 {len(val_test_overlap)}개의 중복 파일이 있습니다!")
        overlap_found = True
    else:
        print("✅ Val ∩ Test: 중복 없음")

    if not overlap_found:
        print("\n✨ 모든 데이터셋이 완벽하게 분리되었습니다 (Leakage 없음).")
    
    # 3. Compare the total count with the paper.
    print("\n📉 수치 검증")
    print(f"   - 현재 총 파일 수: {total_files:,}")
    print(f"   - 논문 언급(Total Frames): 약 {EXPECTED_TOTAL_FRAMES_APPROX:,}")
    
    diff = abs(total_files - EXPECTED_TOTAL_FRAMES_APPROX)
    if diff < 5000: # Allow a margin of about 5,000 for dataset-version differences.
        print("✅ 파일 개수가 논문 수치와 대체로 일치합니다.")
    else:
        print(f"⚠️ 참고: 논문 수치와 차이가 큽니다 ({diff:,} 개 차이).")
        print("   (TestTeal이나 원본 데이터의 일부 누락/추가 여부를 확인해 보세요.)")

if __name__ == '__main__':
    if os.path.exists(DATA_ROOT):
        check_integrity()
    else:
        print(f"🚨 데이터 경로를 찾을 수 없습니다: {DATA_ROOT}")
        print("1_convert_data.py를 먼저 실행했는지 확인하세요.")
