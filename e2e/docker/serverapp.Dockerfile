FROM flwr/serverapp:nightly

WORKDIR /app

COPY server.py ./
ENTRYPOINT [ "flower-server-app", "server:app" ]
