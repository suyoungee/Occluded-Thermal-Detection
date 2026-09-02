# /scripts/2_train.py

import os
from ultralytics import YOLO

# Define configuration paths.
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_CONFIG = os.path.join(PROJECT_ROOT, 'configs', 'roboflow_data.yaml')
MODEL_TYPE = 'yolo11m.pt' # Pretrained YOLOv11m weights.

def train_model():
    # 1. Load the YOLO model with pretrained weights.
    # The weights are downloaded automatically when the .pt file is missing.
    model = YOLO(MODEL_TYPE) 
    
    print(f"YOLOv11 모델 ({MODEL_TYPE}) 학습 시작...")
    
    # 2. Fine-tune the model.
    model.train(
        data=DATA_CONFIG,
        epochs=80,           # Number of training epochs.
        imgsz=640,            # Input image size.
        device=[0,1],
        batch=16,             # Adjust the batch size to available GPU memory.
        lr0=0.0001,              # Initial learning rate.
        project='runs/detect',# Directory for training outputs.
        name='train', # Run directory name.
        workers=8,            # Number of data-loading workers.
    )
    
    print("학습 완료.")

if __name__ == '__main__':
    train_model()
