# Copyright 2023 Adap GmbH. All Rights Reserved.

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND noninteractive

# Install system dependencies
RUN apt-get update \
    && apt-get -y --no-install-recommends install \
    clang-format git unzip ca-certificates openssh-client liblzma-dev \
    build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev wget\
    libsqlite3-dev curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev \
    libxmlsec1-dev libffi-dev liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# Install PyEnv and Python
ARG PYTHON_VERSION=3.10.9
ENV PYENV_ROOT /root/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
RUN pyenv install ${PYTHON_VERSION} \
    && pyenv global ${PYTHON_VERSION} \
    && pyenv rehash

# Install general Python dependencies
ARG POETRY_VERSION=1.5.1
RUN curl -sSL https://install.python-poetry.org | python3 - --version ${POETRY_VERSION}
ENV PATH /root/.local/bin:$PATH
RUN poetry config virtualenvs.create false

# Define default entry point
# ENTRYPOINT [ "poetry", "run", "python", "-m" ]
