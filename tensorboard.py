
import json
import os
import pandas as pd
from tensorboardX import SummaryWriter

# TensorBoardX SummaryWriter 설정

# TensorBoardX SummaryWriter 설정
writer = SummaryWriter('runs/yolov8')

# CSV 파일 불러오기
results = pd.read_csv('/content/results.csv')

# 열 이름의 공백 제거
results.columns = results.columns.str.strip()

# CSV 파일의 열 이름 출력 (공백 제거 확인)
print(results.columns)

# 각 에포크에 대한 손실 값 및 메트릭 기록
for index, row in results.iterrows():
    epoch = index + 1
    
    writer.add_scalars('Loss/box_loss', {'train': row['train/box_loss'], 'val': row['val/box_loss']}, epoch)
    
    # Class Loss 값 기록 (하나의 그래프로 통합)
    writer.add_scalars('Loss/cls_loss', {'train': row['train/cls_loss'], 'val': row['val/cls_loss']}, epoch)
    
    # DFL Loss 값 기록 (하나의 그래프로 통합)
    writer.add_scalars('Loss/dfl_loss', {'train': row['train/dfl_loss'], 'val': row['val/dfl_loss']}, epoch)
    
    # 메트릭 기록
    writer.add_scalar('Metrics/precision', row['metrics/precision(B)'], epoch)
    writer.add_scalar('Metrics/recall', row['metrics/recall(B)'], epoch)
    writer.add_scalar('Metrics/mAP50', row['metrics/mAP50(B)'], epoch)
    writer.add_scalar('Metrics/mAP50-95', row['metrics/mAP50-95(B)'], epoch)

    # 학습률 기록
    writer.add_scalar('LearningRate/pg0', row['lr/pg0'], epoch)
    writer.add_scalar('LearningRate/pg1', row['lr/pg1'], epoch)
    writer.add_scalar('LearningRate/pg2', row['lr/pg2'], epoch)

# TensorBoardX writer 닫기
writer.close()

# TensorBoard 실행
%load_ext tensorboard
%tensorboard --logdir=runs/yolov8
