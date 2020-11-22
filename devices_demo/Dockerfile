ARG BASE_IMAGE_TYPE=cpu
FROM jafermarq/jetsonfederated_$BASE_IMAGE_TYPE:latest

WORKDIR /app

ADD client.py /app
ADD utils.py /app

RUN echo "Ready!"
ENTRYPOINT ["python3","./client.py"]