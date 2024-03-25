# Copyright 2023 Flower Labs GmbH. All Rights Reserved.

ARG PYTHON_VERSION=3.10
ARG ALPINE_VERSION=alpine3.19
FROM python:${PYTHON_VERSION}-${ALPINE_VERSION} as base

ENV DEBIAN_FRONTEND=noninteractive \
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

# Install system dependencies
RUN apk add --no-cache \
  git=2.43.0-r0 \
  py3-pip=23.3.1-r0

# Install specific version of pip
ARG PIP_VERSION
ARG SETUPTOOLS_VERSION
RUN pip install --no-cache-dir \
    pip==${PIP_VERSION:-24.0} \
    setuptools==${SETUPTOOLS_VERSION:-69.2.0} \
    # Add User
    && adduser -D -u 49999 app \
    && mkdir -p /app \
    && chown -R app:app /app

WORKDIR /app
USER app
ENV HOME=/app
