ARG BASE_IMAGE_TYPE=cpu
# these images have been pushed to Dockerhub but you can find
# each Dockerfile used in the `base_images` directory 
FROM jafermarq/jetsonfederated_$BASE_IMAGE_TYPE:latest

WORKDIR /app

ADD client.py /app
ADD utils.py /app

RUN echo "Ready!"
ENTRYPOINT ["python3","./client.py"]