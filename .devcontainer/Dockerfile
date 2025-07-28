# .devcontainer/Dockerfile
FROM python:3.11-slim

# 1) 시스템 의존성 설치 (필요시 수정)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       git build-essential libglib2.0-0  libglib2.0-dev \
       libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 2) 작업 디렉토리 생성
WORKDIR /workspace

# 3) requirements 복사 및 패키지 설치
COPY requirements.txt requirements-dev.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install -r requirements-dev.txt \
    && pip install notebook

# 4) 포트 열기 & 기본 커맨드
EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''"]

