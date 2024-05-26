## Yolov8-30000-epoch3

![image/png](https://cdn-uploads.huggingface.co/production/uploads/664601b5716087aeff3551c9/PVBDoaDw4eioJr7e-iJ0r.png)


![](https://velog.velcdn.com/images/yoose1002/post/f9a82c3d-67e1-4d59-8d76-e772aa06ea21/image.png)
![](https://velog.velcdn.com/images/yoose1002/post/92997ff6-e1b2-4eca-abd1-17a28425a905/image.png)
Loss
![](https://velog.velcdn.com/images/yoose1002/post/24610ff3-8c96-4ad4-ac43-05a757707983/image.png)
![](https://velog.velcdn.com/images/yoose1002/post/20815104-0393-4374-b2d9-324af1a61171/image.png)
![](https://velog.velcdn.com/images/yoose1002/post/12fe0e8a-c7a8-47be-9ea8-6a1728f01770/image.png)
![](https://velog.velcdn.com/images/yoose1002/post/37c3f93a-1009-4411-941b-25eff340001f/image.png)

```
# 이미지 3만장 학습
import torch
from ultralytics import YOLO

# 사전 학습된 YOLOv5 모델 로드
model = YOLO('yolov8n.pt')

# 학습 설정
epochs = 3
data_path = 'data/data.yaml'
img_size = 640
batch_size = 10
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
```
