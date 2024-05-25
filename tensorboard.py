import json
import os
import pandas as pd
from tensorboardX import SummaryWriter

# TensorBoardX SummaryWriter 설정
writer = SummaryWriter('runs/yolov5_loss_results')

# 평가 결과가 저장된 디렉토리와 파일
results_dir = 'detect'
results_file = os.path.join(results_dir, 'results.csv')


# CSV 파일 불러오기
results = pd.read_csv(results_file)

# 열 이름의 공백 제거
results.columns = results.columns.str.strip()

# CSV 파일의 열 이름 출력 (공백 제거 확인)
print(results.columns)

# 각 에포크에 대한 손실 값 및 메트릭 기록
for index, row in results.iterrows():
    epoch = index + 1
    
    # 학습 손실 값 기록
    writer.add_scalar('Loss/train/box_loss', row['train/box_loss'], epoch)
    writer.add_scalar('Loss/train/cls_loss', row['train/cls_loss'], epoch)
    writer.add_scalar('Loss/train/dfl_loss', row['train/dfl_loss'], epoch)
    
    # 검증 손실 값 기록
    writer.add_scalar('Loss/val/box_loss', row['val/box_loss'], epoch)
    writer.add_scalar('Loss/val/cls_loss', row['val/cls_loss'], epoch)
    writer.add_scalar('Loss/val/dfl_loss', row['val/dfl_loss'], epoch)
    
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
%tensorboard --logdir=runs/yolov5_loss_results
