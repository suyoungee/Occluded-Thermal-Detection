# /scripts/4_test.py

import os
from ultralytics import YOLO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_CONFIG = os.path.join(PROJECT_ROOT, 'configs', 'roboflow_data.yaml')

# Path to the weights produced by training.
SELECT_WEIGHTS = 'train8'
BEST_WEIGHTS = os.path.join(PROJECT_ROOT, 'runs', 'detect', SELECT_WEIGHTS, 'weights', 'best.pt')

def test_model_quantitatively():
    """
    Run quantitative evaluation on the test dataset
    defined by configs/roboflow_data.yaml (split='test').
    """
    # 1. Load the trained model.
    if not os.path.exists(BEST_WEIGHTS):
        print(f"🚨 오류: 모델 가중치 파일을 찾을 수 없습니다: {BEST_WEIGHTS}")
        model = YOLO('yolo11m.pt')  # Load the default pretrained model.
        model_name = 'yolov11m.pt (Pre-trained Baseline)'
        print("기본 사전 학습된 YOLOv11m 모델을 로드합니다.")
        
    else:
        model = YOLO(BEST_WEIGHTS)
        model_name = os.path.basename(BEST_WEIGHTS)
        print(f"{BEST_WEIGHTS}모델을 로드합니다.")
    
    print(f"모델 ({model_name}) 테스트셋 정량 평가 시작...")
    
    # 2. Run quantitative evaluation through Ultralytics model.val().
    metrics = model.val(
        data=DATA_CONFIG,
        imgsz=640,
        batch=32,
        split='test' # Explicitly select the test split.
    )

    # Print quantitative metrics.
    print(f"\n--- ROBOFLOW 테스트셋 정량 평가 결과 ---")
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
    print(f"Precision: {metrics.box.mp}")
    print(f"Recall: {metrics.box.mr}")
    print("--------------------------------------")

if __name__ == '__main__':
    test_model_quantitatively()
