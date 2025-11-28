# /scripts/5_predict.py

import os
import glob
from ultralytics import YOLO
import shutil
import datetime

# -------------------------- 설정 --------------------------
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
BEST_WEIGHTS = ""

# 추론할 이미지 소스 폴더
SOURCE_PATH = os.path.join(PROJECT_ROOT, 'assets') 

# 최종 결과 이미지를 저장할 폴더 (results)
OUTPUT_RESULT_DIR = os.path.join(PROJECT_ROOT, 'results')

# Ultralytics가 추론 결과를 임시로 저장하는 기본 폴더 경로
TEMP_PREDICT_DIR_ROOT = os.path.join(PROJECT_ROOT, 'runs', 'predict')
# ----------------------------------------------------------

def predict_on_assets_for_qualitative_assessment():
    """
    assets 폴더 내의 임의 이미지에 대해 정성적(시각적) 추론을 실행하고, 
    결과 이미지를 'results' 폴더로 이동 후 임시 폴더를 정리합니다.
    """
    
    # 결과 폴더 생성
    os.makedirs(OUTPUT_RESULT_DIR, exist_ok=True)
    
    # 1. 모델 가중치 확인 및 로드
    if not os.path.exists(BEST_WEIGHTS):
        print(f"🚨 오류: 모델 가중치 파일을 찾을 수 없습니다: {BEST_WEIGHTS}")
        model = YOLO('yolo11m.pt')  # 기본 사전 학습된 모델 로드
        print("기본 사전 학습된 YOLOv11m 모델을 로드합니다.")
        
    else:
        model = YOLO(BEST_WEIGHTS)
    
    # 2. 임시 추론 폴더 이름 생성
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_folder_name = f'temp_qualitative_{timestamp}'
    
    print(f"모델 ({os.path.basename(BEST_WEIGHTS)})을 사용하여 assets 폴더 추론 시작 (정성적 평가)...")
    
    # 3. 추론 실행
    model.predict(
        source=SOURCE_PATH,
        conf=0.25,        # 신뢰도 임계값
        imgsz=640,
        save=True,        # 결과를 디스크에 저장
        project=TEMP_PREDICT_DIR_ROOT,
        name=temp_folder_name,
        exist_ok=True,
        verbose=False
    )
    
    # 4. 결과 파일 이동 및 정리
    temp_result_path = os.path.join(TEMP_PREDICT_DIR_ROOT, temp_folder_name)
    
    if os.path.exists(temp_result_path):
        print(f"추론 결과 파일을 최종 저장 폴더 ({os.path.basename(OUTPUT_RESULT_DIR)})로 이동합니다.")
        
        result_files = glob.glob(os.path.join(temp_result_path, '*.*'))
        
        for file_path in result_files:
            file_name = os.path.basename(file_path)
            # 결과 파일명에 '_pred' 접미사를 추가
            base, ext = os.path.splitext(file_name)
            new_file_name = f"{base}_pred{ext}"
            
            # 최종 목적지 (results 폴더)
            destination_path = os.path.join(OUTPUT_RESULT_DIR, new_file_name)
            
            shutil.move(file_path, destination_path)
            print(f"-> {new_file_name} 저장 완료")
            
        # 임시 추론 폴더 삭제
        shutil.rmtree(temp_result_path)
        print("임시 추론 폴더 정리 완료.")
    else:
        print("경고: 추론 결과 폴더를 찾을 수 없습니다.")

if __name__ == '__main__':
    predict_on_assets_for_qualitative_assessment()