ARG BASE_IMAGE

# Pull the base image from NVIDIA
FROM $BASE_IMAGE

# Update pip
RUN pip3 install --upgrade pip

# Install flower
RUN pip3 install flwr>=1.0
RUN pip3 install flwr-datsets>=0.0.2
RUN pip3 install tqdm==4.65.0

WORKDIR /client
