# /scripts/5_predict.py

import os
import glob
from ultralytics import YOLO
import shutil
import datetime

# ----------------------- Configuration -----------------------
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
BEST_WEIGHTS = ""

# Directory containing source images for inference.
SOURCE_PATH = os.path.join(PROJECT_ROOT, 'assets') 

# Final directory for prediction images.
OUTPUT_RESULT_DIR = os.path.join(PROJECT_ROOT, 'results')

# Root directory used by Ultralytics for temporary prediction outputs.
TEMP_PREDICT_DIR_ROOT = os.path.join(PROJECT_ROOT, 'runs', 'predict')
# ----------------------------------------------------------

def predict_on_assets_for_qualitative_assessment():
    """
    Run qualitative inference on images in the assets directory, move the
    results to the results directory, and remove the temporary directory.
    """
    
    # Create the final results directory.
    os.makedirs(OUTPUT_RESULT_DIR, exist_ok=True)
    
    # 1. Check and load model weights.
    if not os.path.exists(BEST_WEIGHTS):
        print(f"🚨 오류: 모델 가중치 파일을 찾을 수 없습니다: {BEST_WEIGHTS}")
        model = YOLO('yolo11m.pt')  # Load the default pretrained model.
        print("기본 사전 학습된 YOLOv11m 모델을 로드합니다.")
        
    else:
        model = YOLO(BEST_WEIGHTS)
    
    # 2. Create a unique temporary prediction directory.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_folder_name = f'temp_qualitative_{timestamp}'
    
    print(f"모델 ({os.path.basename(BEST_WEIGHTS)})을 사용하여 assets 폴더 추론 시작 (정성적 평가)...")
    
    # 3. Run inference.
    model.predict(
        source=SOURCE_PATH,
        conf=0.25,        # Confidence threshold.
        imgsz=640,
        save=True,        # Save predictions to disk.
        project=TEMP_PREDICT_DIR_ROOT,
        name=temp_folder_name,
        exist_ok=True,
        verbose=False
    )
    
    # 4. Move result files and clean up.
    temp_result_path = os.path.join(TEMP_PREDICT_DIR_ROOT, temp_folder_name)
    
    if os.path.exists(temp_result_path):
        print(f"추론 결과 파일을 최종 저장 폴더 ({os.path.basename(OUTPUT_RESULT_DIR)})로 이동합니다.")
        
        result_files = glob.glob(os.path.join(temp_result_path, '*.*'))
        
        for file_path in result_files:
            file_name = os.path.basename(file_path)
            # Append the '_pred' suffix to the result filename.
            base, ext = os.path.splitext(file_name)
            new_file_name = f"{base}_pred{ext}"
            
            # Final destination in the results directory.
            destination_path = os.path.join(OUTPUT_RESULT_DIR, new_file_name)
            
            shutil.move(file_path, destination_path)
            print(f"-> {new_file_name} 저장 완료")
            
        # Remove the temporary prediction directory.
        shutil.rmtree(temp_result_path)
        print("임시 추론 폴더 정리 완료.")
    else:
        print("경고: 추론 결과 폴더를 찾을 수 없습니다.")

if __name__ == '__main__':
    predict_on_assets_for_qualitative_assessment()
