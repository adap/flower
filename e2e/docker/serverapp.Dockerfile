FROM flwr/serverapp:1.9.0.dev20240513

WORKDIR /app

COPY server.py ./
ENTRYPOINT [ "flower-server-app", "server:app" ]
