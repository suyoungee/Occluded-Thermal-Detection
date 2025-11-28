# /scripts/4_test.py

import os
from ultralytics import YOLO

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_CONFIG = os.path.join(PROJECT_ROOT, 'configs', 'roboflow_data.yaml')

# 학습 결과가 저장된 weights 경로를 지정합니다.
BEST_WEIGHTS = '/home/suyoung/study/IPIU-2026/runs/detect/train8/weights/best.pt'

def test_model_quantitatively():
    """
    테스트 데이터셋(data/BIRDSAI/images/test)에 대해 정량적 평가를 실행합니다.
    """
    # 1. 학습된 모델 로드
    if not os.path.exists(BEST_WEIGHTS):
        print(f"🚨 오류: 모델 가중치 파일을 찾을 수 없습니다: {BEST_WEIGHTS}")
        model = YOLO('yolo11m.pt')  # 기본 사전 학습된 모델 로드
        model_name = 'yolov11m.pt (Pre-trained Baseline)'
        print("기본 사전 학습된 YOLOv11m 모델을 로드합니다.")
        
    else:
        model = YOLO(BEST_WEIGHTS)
        model_name = os.path.basename(BEST_WEIGHTS)
        print(f"{BEST_WEIGHTS}모델을 로드합니다.")
    
    print(f"모델 ({model_name}) 테스트셋 정량 평가 시작...")
    
    # 2. 테스트 실행 (Ultralytics는 model.val() 함수를 통해 정량적 평가를 수행합니다.)
    # data/BIRDSAI/images/test 경로에 있는 데이터를 사용하도록 설정됩니다.
    metrics = model.val(
        data=DATA_CONFIG,
        imgsz=640,
        batch=32,
        split='test' # <--- 테스트 셋을 명시적으로 지정
        # save_json=True  # COCO 형식으로 결과를 저장하려면 활성화
    )

    # 정량적 결과 출력
    print(f"\n--- ROBOFLOW 테스트셋 정량 평가 결과 ---")
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
    print(f"Precision: {metrics.box.mp}")
    print(f"Recall: {metrics.box.mr}")
    print("--------------------------------------")

if __name__ == '__main__':
    test_model_quantitatively()