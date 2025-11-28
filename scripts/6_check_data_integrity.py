import os
import glob

# ----------------- 설정 -----------------
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'RoboFlow_FLIR_Dataset_v27_yolo11', 'images')

# 논문 기준 예상 프레임 수 (참고용)
# Real: ~62k, Sim: ~100k -> Total: ~162k
# (정확한 숫자는 원본 데이터셋 버전에 따라 약간 다를 수 있으므로 근사치로 확인)
EXPECTED_TOTAL_FRAMES_APPROX = 162000 

def check_integrity():
    print("🧐 데이터셋 무결성 검사를 시작합니다...\n")
    
    splits = ['train', 'val', 'test']
    file_sets = {}
    total_files = 0
    
    # 1. 각 Split별 파일 수집
    for split in splits:
        split_path = os.path.join(DATA_ROOT, split)
        
        if not os.path.exists(split_path):
            print(f"🚨 경고: {split} 폴더가 존재하지 않습니다! ({split_path})")
            file_sets[split] = set()
            continue
            
        # 이미지 파일 목록 가져오기
        files = glob.glob(os.path.join(split_path, '*.jpg'))
        # 경로 제외하고 파일명만 추출
        filenames = [os.path.basename(f) for f in files]
        
        file_sets[split] = set(filenames)
        count = len(filenames)
        total_files += count
        
        print(f"📁 {split.upper()} 셋 파일 수: {count:,} 개")

    print(f"\n📊 총 이미지 파일 수: {total_files:,} 개")
    
    # 2. 중복 검사 (Mutual Exclusivity)
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
    
    # 3. 총 개수 검증 (논문 수치와 비교)
    print("\n📉 수치 검증")
    print(f"   - 현재 총 파일 수: {total_files:,}")
    print(f"   - 논문 언급(Total Frames): 약 {EXPECTED_TOTAL_FRAMES_APPROX:,}")
    
    diff = abs(total_files - EXPECTED_TOTAL_FRAMES_APPROX)
    if diff < 5000: # 오차 범위 5000개 내외면 정상으로 간주 (데이터셋 버전에 따라 다를 수 있음)
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