# Copyright 2023 Flower Labs GmbH. All Rights Reserved.

ARG BASE_IMAGE_VERSION=py3.11-ubuntu22.04
ARG BASE_REPOSITORY=flwr/base
FROM $BASE_REPOSITORY:$BASE_IMAGE_VERSION as server

WORKDIR /app
ARG FLWR_VERSION
RUN python -m pip install -U --no-cache-dir flwr[rest]==${FLWR_VERSION}
ENTRYPOINT ["python", "-c", "from flwr.server import run_server; run_server()"]

# Test if Flower can be successfully installed and imported
FROM server as test
RUN python -c "from flwr.server import run_server"
