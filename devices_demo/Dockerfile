FROM jafermarq/jetsonfederated_cpu:test


WORKDIR /app

ADD client.py /app
ADD utils.py /app

RUN echo "Ready!"
ENTRYPOINT ["python3","./client.py"]