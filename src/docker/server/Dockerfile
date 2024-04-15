# Copyright 2023 Flower Labs GmbH. All Rights Reserved.

ARG BASE_REPOSITORY=flwr/base
ARG PYTHON_VERSION=3.10
ARG ALPINE_VERSION=alpine3.19
FROM ${BASE_REPOSITORY}:py${PYTHON_VERSION}-${ALPINE_VERSION} as server

ARG FLWR_VERSION
RUN pip install -U --no-cache-dir \
    flwr[rest]==${FLWR_VERSION:-1.7.0}

ENTRYPOINT ["flower-server"]

# Test if Flower can be successfully installed and imported
FROM server as test
RUN python -c "from flwr.server import run_superlink"
