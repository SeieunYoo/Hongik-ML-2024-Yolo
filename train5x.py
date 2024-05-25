from ultralytics import YOLO
from datasets import load_dataset

model = YOLO('yolov5x.pt')

results = model.train(
   data='data/data.yaml',
   imgsz=1024,
   epochs=1,
   batch=4,
   name='yolov5x_custom',
   lr0=0.01,  # 초기 학습률
   lrf=0.01,  # 최종 학습률 (lr0 * lrf)
   momentum=0.937,  # 모멘텀
   weight_decay=0.0005,  # 옵티마이저 가중치 감쇠
   warmup_epochs=1.0,  # 워밍업 에포크 수 (기존 3.0 -> 1.0)
   warmup_momentum=0.8,  # 워밍업 초기 모멘텀
   warmup_bias_lr=0.1,  # 워밍업 초기 바이어스 학습률
   box=0.02,  # 박스 손실 가중치 (기존 0.05 -> 0.02)
   cls=0.5,  # 클래스 손실 가중치
   kobj=1.0,  # 객체 BCELoss positive_weight
   iou=0.2,  # IoU 학습 임계값
   hsv_h=0.015,  # 이미지 HSV-Hue 보강 (fraction)
   hsv_s=0.7,  # 이미지 HSV-채도 보강 (fraction)
   hsv_v=0.4,  # 이미지 HSV-밝기 보강 (fraction)
   degrees=0.0,  # 이미지 회전 (+/- deg)
   translate=0.1,  # 이미지 이동 (+/- fraction)
   scale=0.5,  # 이미지 스케일 (+/- gain)
   shear=0.0,  # 이미지 기울이기 (+/- deg)
   perspective=0.0,  # 이미지 원근 보강 (+/- fraction), 범위 0-0.001
   flipud=0.0,  # 이미지 상하 반전 (확률)
   fliplr=0.5,  # 이미지 좌우 반전 (확률)
   mosaic=0.5,  # 이미지 모자이크 (확률) (기존 1.0 -> 0.5)
   mixup=0.0,  # 이미지 믹스업 (확률)
   copy_paste=0.0  # 세그먼트 복사-붙여넣기 (확률)
)
