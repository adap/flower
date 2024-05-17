FROM flwr/supernode:nightly

WORKDIR /app
COPY pyproject.toml ./
RUN python -m pip install -U --no-cache-dir . \
    && pyenv rehash

COPY client.py ./
ENTRYPOINT [ "flower-client-app", "client:app" ]
