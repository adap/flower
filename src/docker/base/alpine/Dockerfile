# Copyright 2024 Flower Labs GmbH. All Rights Reserved.

# Multi-stage base image build
#
# Building the base image for ARM64 requires installing some additional system dependencies to
# compile the grpcio package, as they do not provide a pre-built package. However, we don't want
# the dependencies in the final base image as they are only needed to compile the package.
# That's why we're creating a multi-stage build. When installing the flwr, we are using a
# virtual environment to keep all files in a single isolated directory as described here:
# https://pythonspeed.com/articles/multi-stage-docker-python/

ARG PYTHON_VERSION=3.11
ARG OS_VERSION=alpine3.19
FROM python:${PYTHON_VERSION}-${OS_VERSION} as compile

# Install system dependencies
RUN apk add --no-cache \
    # require for compiling grpcio on ARM64
    g++=13.2.1_git20231014-r0 \
    libffi-dev=3.4.4-r3 \
    # create virtual env
    && python -m venv /opt/venv

# Make sure we use the virtualenv
ENV PATH="/opt/venv/bin:$PATH"

# Install specific version of pip, setuptools and flwr
ARG PIP_VERSION
ARG SETUPTOOLS_VERSION
ARG FLWR_VERSION
ARG FLWR_PACKAGE=flwr
RUN pip install -U --no-cache-dir \
    pip==${PIP_VERSION} \
    setuptools==${SETUPTOOLS_VERSION} \
    ${FLWR_PACKAGE}==${FLWR_VERSION}

FROM python:${PYTHON_VERSION}-${OS_VERSION} as base

# required by the grpc package
RUN apk add --no-cache \
    libstdc++=13.2.1_git20231014-r0

COPY --from=compile /opt/venv /opt/venv

# Make sure we use the virtualenv
ENV ENV PATH="/opt/venv/bin:$PATH" \
    # Send stdout and stderr stream directly to the terminal. Ensures that no
    # output is retained in a buffer if the application crashes.
    PYTHONUNBUFFERED=1 \
    # Typically, bytecode is created on the first invocation to speed up following invocation.
    # However, in Docker we only make a single invocation (when we start the container).
    # Therefore, we can disable bytecode writing.
    PYTHONDONTWRITEBYTECODE=1 \
    # Ensure that python encoding is always UTF-8.
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# add non-root user
RUN adduser \
    --no-create-home \
    --disabled-password \
    --gecos "" \
    --uid 49999 app \
    && mkdir -p /app \
    && chown -R app:app /app

WORKDIR /app
USER app
ENV HOME=/app
