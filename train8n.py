import torch
from tensorboardX import SummaryWriter
from ultralytics import YOLO

# TensorBoardX SummaryWriter 설정
writer = SummaryWriter('runs/yolov8_example')

# 사전 학습된 YOLOv5 모델 로드
model = YOLO('yolov8n.pt')

# 학습 설정
epochs = 3
data_path = 'data/data.yaml'
img_size = 640
batch_size = 4
project_name = 'yolov8n_custom'
project_dir = 'runs/train'

# 사용자 정의 학습 함수
def train_yolov8(model, data_path, img_size, batch_size, epochs, project_name, project_dir):
    results = model.train(
        data=data_path,
        imgsz=img_size,
        epochs=epochs,
        batch=batch_size,
        name=project_name,
        project=project_dir,
        warmup_epochs=1.0,
        box=0.02,
        mosaic=0.5,  
    )

# 모델 학습
train_yolov8(model, data_path, img_size, batch_size, epochs, project_name, project_dir)
