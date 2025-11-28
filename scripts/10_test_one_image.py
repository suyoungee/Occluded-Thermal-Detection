import os
from ultralytics import YOLO

# ------------------- 설정 -------------------
# 1. 테스트할 이미지 경로 (저장하신 경로)
IMAGE_PATH = '/home/suyoung/study/IPIU-2026/data/RoboFlow_FLIR_Dataset_v27_yolo11/train/images/car_99-94469285011292_Sat-Nov-27-12-11-12-2021_jpg.rf.2ee14e60e40ef96503d665a3175639ce.jpg'

# 2. 학습된 모델 가중치 경로
# (주의: 학습이 여러 번 돌아갔으면 train, train2, train3 등 폴더명이 다를 수 있습니다.
#  가장 최근에 학습된 폴더의 best.pt를 지정하세요.)
MODEL_PATH = 'runs/detect/train8/weights/best.pt' 
# -------------------------------------------

def test_single_image():
    # 이미지 파일 존재 여부 확인
    if not os.path.exists(IMAGE_PATH):
        print(f"🚨 오류: 이미지를 찾을 수 없습니다 -> {IMAGE_PATH}")
        return

    # 모델 가중치 존재 여부 확인
    if not os.path.exists(MODEL_PATH):
        print(f"🚨 오류: 모델 파일을 찾을 수 없습니다 -> {MODEL_PATH}")
        print("경로를 다시 확인하거나, 학습이 완료되었는지 확인하세요.")
        return

    print(f"🚀 모델 로딩 중... ({MODEL_PATH})")
    model = YOLO(MODEL_PATH)

    print(f"📸 이미지 추론 시작... ({os.path.basename(IMAGE_PATH)})")
    
    # 추론 실행
    # save=True: 결과 이미지를 디스크에 저장
    # conf=0.25: 신뢰도가 0.25 이상인 것만 표시 (기본값)
    results = model.predict(
        source=IMAGE_PATH,
        save=True,
        imgsz=640,
        conf=0.25, 
        project='runs/predict', # 결과 저장 루트
        name='occlusion_test',  # 결과 저장 폴더 이름
        exist_ok=True           # 덮어쓰기 허용
    )

    print("\n✅ 테스트 완료!")
    print(f"결과 이미지는 아래 폴더에서 확인하세요:")
    print(f"👉 runs/predict/occlusion_test/")

if __name__ == '__main__':
    test_single_image()