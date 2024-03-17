# Copyright 2023 Flower Labs GmbH. All Rights Reserved.

ARG BASE_REPOSITORY=flwr/base
ARG PYTHON_VERSION=3.10
ARG ALPINE_VERSION=alpine3.19
FROM ${BASE_REPOSITORY}:py${PYTHON_VERSION}-${ALPINE_VERSION} as client

ARG FLWR_VERSION
RUN pip install -U --no-cache-dir \
    flwr[rest]==${FLWR_VERSION:-1.7.0}
