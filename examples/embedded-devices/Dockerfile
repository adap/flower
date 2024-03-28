ARG BASE_IMAGE

# Pull the base image from NVIDIA
FROM $BASE_IMAGE

# Update pip && Install flower
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
    flwr>=1.0 \
    flwr-datsets>=0.2 \
    tqdm==4.65.0

WORKDIR /client
