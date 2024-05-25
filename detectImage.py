import cv2
import requests
from ultralytics import YOLO
from IPython.display import Image, display
import os

# YOLO 모델 로드
model = YOLO("/content/best.pt")  # YOLOv8x 모델 가중치 파일 경로

# 이미지 다운로드 함수
def download_image(url, output_path):
    response = requests.get(url)
    with open(output_path, 'wb') as file:
        file.write(response.content)

# 이미지 감지 및 저장 함수
def detect_image(image_path, output_path):
    img = cv2.imread(image_path)
    results = model(img)
    annotated_img = results[0].plot()
    cv2.imwrite(output_path, annotated_img)

# 이미지 URL
image_url = 'https://github.com/SeieunYoo/SeieunYoo/assets/101736358/e423c809-cc4d-41b8-8efb-030f3e47d7b3'  # 실제 이미지 URL로 변경

# 이미지 다운로드
image_path = 'downloaded_image.jpg'
download_image(image_url, image_path)

# 이미지 감지 및 결과 저장
output_image_path = 'detected_image.jpg'
detect_image(image_path, output_image_path)

# 결과 이미지 표시
display(Image(output_image_path))
