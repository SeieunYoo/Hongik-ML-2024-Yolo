# 필요한 라이브러리 설치
!pip install huggingface_hub

# Hugging Face Hub 로그인
from huggingface_hub import notebook_login

notebook_login()

# Git LFS 설치 (Colab 환경)
!apt-get install git-lfs
!git lfs install

# Hugging Face Hub에 리포지토리 생성 및 설정
from huggingface_hub import HfApi

api = HfApi()
username = api.whoami()['name']
repo_name = "Yolov8n-30000-3"
repo_id = f"{username}/{repo_name}"

# 리포지토리 생성
api.create_repo(repo_id=repo_id, exist_ok=True)

# Colab 환경에서 작업 디렉토리 설정
repo_local_dir = f"/content/{repo_name}"

# 로컬 디렉토리 초기화 및 학습 결과 복사
import os
import shutil

# 디렉토리 초기화
!mkdir -p {repo_local_dir}
!cd {repo_local_dir} && git init

# 사용자 이름과 이메일 설정
!git config --global user.email "your_email@example.com"
!git config --global user.name "Your Name"

# 기존 원격 리포지토리 제거 (이미 존재할 경우)
!cd {repo_local_dir} && git remote remove origin

# 새로운 원격 리포지토리 추가
!cd {repo_local_dir} && git remote add origin https://huggingface.co/{repo_id}

# 학습 결과 디렉토리
source_dir = "runs/train/exp"
destination_dir = repo_local_dir

if os.path.exists(destination_dir):
    shutil.rmtree(destination_dir)
shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)

# 리포지토리 디렉토리로 이동
%cd {repo_local_dir}

# 새로운 브랜치 생성 및 체크아웃
!git checkout -b main

# 원격 브랜치에서 변경 사항 가져오기 및 병합
!git pull origin main --rebase

# 파일 추가 및 커밋
!git add .
!git commit -m "Add YOLOv8 model training results"

# Hugging Face Hub에 푸시
!git push origin main

print(f"Model uploaded to https://huggingface.co/{repo_id}")
