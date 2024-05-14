FROM flwr/supernode:1.9.0.dev20240513

WORKDIR /app
RUN python -m pip install -U --no-cache-dir\
    "flwr-datasets["vision"]>=0.1.0,<1.0.0" \
    torch==2.2.1 \
    torchvision==0.17.1 \
    && pyenv rehash

COPY client.py ./
ENTRYPOINT [ "flower-client-app", "client:app" ]
