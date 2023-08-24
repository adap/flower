# Copyright 2023 Flower Labs GmbH. All Rights Reserved.

FROM ubuntu:22.04 as build

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
ARG PYTHON_VERSION
ENV PYENV_ROOT /root/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
RUN pyenv install ${PYTHON_VERSION} \
    && pyenv global ${PYTHON_VERSION} \
    && pyenv rehash

# Install more recent version of setuptools
RUN python -m pip install setuptools==68.1.2

# Install poetry as all examples use it and therefore it should be available for custom images
ARG POETRY_VERSION=1.5.1
RUN curl -sSL https://install.python-poetry.org | python3 - --version ${POETRY_VERSION}
ENV PATH /root/.local/bin:$PATH
RUN poetry config virtualenvs.create false

# Test if Flower can be successfully installed and imported
# Do not use poetry as that would require a pyproject.toml
FROM build as test
RUN python -m pip install flwr
RUN python -c "from flwr.server import run_server"
