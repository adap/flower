# Uses a pre-built image from nvidia, then we install Flower
# We also install dependencies for a very exciting future project

# check this repo to learn more about the image below: https://github.com/dusty-nv/jetson-containers
FROM nvcr.io/nvidia/l4t-pytorch:r32.4.4-pth1.6-py3

RUN mkdir /app
WORKDIR /app

# update pip
RUN python3 -m pip install --upgrade pip

# Installing flower
RUN pip3 install flwr==0.13.0

# Deps for an exciting future project
RUN apt-get update
RUN apt-get install libffi-dev python-dev python3-h5py libhdf5-dev libssl-dev \
    gcc gfortran libopenblas-dev liblapack-dev cython -y
RUN pip3 install scikit-build

RUN pip3 install cmake joblib numpy pre-commit psutil==5.6.6 \
    pyyaml ruamel.yaml scipy sentencepiece SoundFile \
    sphinx-rtd-theme==0.4.3 Sphinx tqdm

RUN echo "done!"
