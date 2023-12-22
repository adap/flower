# Copyright 2023 Flower Labs GmbH. All Rights Reserved.

ARG BASE_REPOSITORY=flwr/base
ARG BASE_IMAGE_TAG
FROM $BASE_REPOSITORY:$BASE_IMAGE_TAG

ARG FLWR_VERSION
RUN python -m pip install -U --no-cache-dir flwr[rest]==${FLWR_VERSION}
