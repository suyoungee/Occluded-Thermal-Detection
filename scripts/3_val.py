# /scripts/3_val.py

import os
from ultralytics import YOLO

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_CONFIG = os.path.join(PROJECT_ROOT, 'configs', 'roboflow_data.yaml')

# Path to the weights produced by training.
# A typical path is 'runs/detect/yolov11m_birdsai_occlusion_ft/weights/best.pt'.
BEST_WEIGHTS = '' 

def validate_model():
    # 1. Load the trained model.
    if not os.path.exists(BEST_WEIGHTS):
        print(f"🚨 오류: 모델 가중치 파일을 찾을 수 없습니다: {BEST_WEIGHTS}")
        model = YOLO('yolo11m.pt')  # Load the default pretrained model.
        model_name = 'yolov11m.pt (Pre-trained Baseline)'
        print("기본 사전 학습된 YOLOv11m 모델을 로드합니다.")
        
    else:
        model = YOLO(BEST_WEIGHTS)
        model_name = os.path.basename(BEST_WEIGHTS)
    
    print(f"모델 ({model_name}) 검증 시작...")
    
    # 2. Run validation.
    metrics = model.val(
        data=DATA_CONFIG,
        imgsz=640,
        batch=32,
        split='val' # Use the validation split.
    )

    # Print validation metrics.
    print(f"\n--- BIRDSAI 검증 결과 ---")
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
    print(f"Precision: {metrics.box.precision}")
    print(f"Recall: {metrics.box.recall}")
    print("--------------------------")

if __name__ == '__main__':
    validate_model()
