# Copyright 2023 Flower Labs GmbH. All Rights Reserved.

ARG BASE_REPOSITORY=flwr/base
ARG BASE_IMAGE_TAG=py3.11-ubuntu22.04
FROM $BASE_REPOSITORY:$BASE_IMAGE_TAG as server

WORKDIR /app
ARG FLWR_VERSION
RUN python -m pip install -U --no-cache-dir flwr[rest]==${FLWR_VERSION}
ENTRYPOINT ["python", "-c", "from flwr.server import run_superlink; run_superlink()"]

# Test if Flower can be successfully installed and imported
FROM server as test
RUN python -c "from flwr.server import run_superlink"
