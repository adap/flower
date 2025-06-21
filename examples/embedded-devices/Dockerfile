# NVIDIA JetPack 6.0 기반 이미지 사용
FROM nvcr.io/nvidia/l4t-base:r36.2.0

# 필수 패키지 업데이트 및 설치
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y python3-pip python3-dev git && \
    rm -rf /var/lib/apt/lists/*

# Python 패키지 업데이트
RUN pip3 install --upgrade pip

# Flower 및 기타 필요한 Python 패키지 설치
RUN pip3 install flwr

RUN pip3 install flwr>=1.0
RUN pip3 install flwr-datasets>=0.0.2
RUN pip3 install tqdm==4.65.0

# 작업 디렉토리 설정
WORKDIR /client