import torch
from tensorboardX import SummaryWriter
from ultralytics import YOLO

# TensorBoardX SummaryWriter 설정
writer = SummaryWriter('runs/yolov8_example')

# 사전 학습된 YOLOv5 모델 로드
model = YOLO('yolov8n.pt')

# 학습 설정
epochs = 10
data_path = 'data/data.yaml'
img_size = 640
batch_size = 4
project_name = 'yolov8n_custom'
project_dir = 'runs/train'

# 사용자 정의 학습 함수
def train_yolov5(model, data_path, img_size, batch_size, epochs, project_name, project_dir):
    results = model.train(
        data=data_path,
        imgsz=img_size,
        epochs=epochs,
        batch=batch_size,
        name=project_name,
        project=project_dir
    )
    metrics = results
    writer.add_scalar('Loss/box', metrics.box_loss, 1)
writer.add_scalar('Loss/obj', metrics.obj_loss, 1)
writer.add_scalar('Loss/cls', metrics.cls_loss, 1)
writer.add_scalar('Loss/total', metrics.total_loss, 1)

writer.add_scalar('Precision', metrics.precision, 1)
writer.add_scalar('Recall', metrics.recall, 1)
writer.add_scalar('mAP@0.5', metrics.map50, 1)
writer.add_scalar('mAP@0.5:0.95', metrics.map, 1)

# TensorBoardX writer 닫기
writer.close()

# 모델 학습
train_yolov5(model, data_path, img_size, batch_size, epochs, project_name, project_dir)

# TensorBoard 실행
%load_ext tensorboard
%tensorboard --logdir=runs/yolov8_example
