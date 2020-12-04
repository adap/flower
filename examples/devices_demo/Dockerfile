ARG BASE_IMAGE_TYPE=cpu
# these images have been pushed to Dockerhub but you can find
# each Dockerfile used in the `base_images` directory 
FROM jafermarq/jetsonfederated_$BASE_IMAGE_TYPE:latest

WORKDIR /app

# Scripts needed for Flower client
ADD client.py /app
ADD utils.py /app

# Download CIFAR-10.
# Having the dataset as part of the Docker image would make things easier in scenarios
# where we want multilple instances of the same image runnig simultaneously.
# If the dataset is already in your system you can `COPY ...` it instead.
ENV DATA_DIR=/app/data/cifar-10
RUN mkdir -p $DATA_DIR
WORKDIR $DATA_DIR
RUN apt-get install wget -y
RUN wget https://www.cs.toronto.edu/\~kriz/cifar-10-python.tar.gz 
RUN tar -zxvf cifar-10-python.tar.gz

RUN echo "Ready!"
WORKDIR /app
ENTRYPOINT ["python3","-u","./client.py"]