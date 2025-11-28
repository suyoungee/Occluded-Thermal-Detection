# /scripts/2_train.py

import os
from ultralytics import YOLO

# 설정 파일 경로 정의
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_CONFIG = os.path.join(PROJECT_ROOT, 'configs', 'roboflow_data.yaml')
MODEL_TYPE = 'yolo11m.pt' # YOLOv11m 사전 학습된 가중치 파일

def train_model():
    # 1. YOLO 모델 로드 (사전 학습된 가중치 사용)
    # .pt 파일이 없으면 자동으로 다운로드됩니다.
    model = YOLO(MODEL_TYPE) 
    
    print(f"YOLOv11 모델 ({MODEL_TYPE}) 학습 시작...")
    
    # 2. 학습 실행 (Fine-tuning)
    results = model.train(
        data=DATA_CONFIG,
        epochs=40,           # 원하는 학습 횟수
        imgsz=640,            # 이미지 크기
        device=[0,1],
        batch=32,             # 배치 사이즈 (GPU 메모리에 따라 조정)
        #patience=100,         # Early stopping patience
        lr0=0.0001,              # 학습률
        project='runs/detect',# 학습 결과 저장 경로
        name='train', # 결과 저장 폴더 이름
        workers=8,            # 데이터 로딩을 위한 워커 수
        # augmentation: BIRDSAI 데이터에 맞춰 Augmentation 설정 조정 가능 (예: hsv, flip 등)
        # cache: True로 설정하여 데이터 로딩 속도 향상
    )
    
    print("학습 완료.")

if __name__ == '__main__':
    train_model()