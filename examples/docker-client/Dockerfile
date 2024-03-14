# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
FROM flwr/client:1.6.0-py3.8-ubuntu22.04

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install -U --no-cache-dir -r requirements.txt

COPY client.py .
ENTRYPOINT ["python", "-c", "from flwr.client import run_client; run_client()", "--callable", "client:flower"]
