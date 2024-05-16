# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

ARG OS_VERSION=22.04
FROM ubuntu:$OS_VERSION as base

ENV DEBIAN_FRONTEND noninteractive
# Send stdout and stderr stream directly to the terminal. Ensures that no
# output is retained in a buffer if the application crashes.
ENV PYTHONUNBUFFERED 1
# Typically, bytecode is created on the first invocation to speed up following invocation.
# However, in Docker we only make a single invocation (when we start the container).
# Therefore, we can disable bytecode writing.
ENV PYTHONDONTWRITEBYTECODE 1
# Ensure that python encoding is always UTF-8.
ENV PYTHONIOENCODING UTF-8
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

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
# https://github.com/hadolint/hadolint/wiki/DL4006
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
RUN pyenv install ${PYTHON_VERSION} \
    && pyenv global ${PYTHON_VERSION} \
    && pyenv rehash

# Install specific version of pip
ARG PIP_VERSION
RUN python -m pip install --no-cache-dir pip==$PIP_VERSION

# Install specific version of setuptools
ARG SETUPTOOLS_VERSION
RUN python -m pip install --no-cache-dir setuptools==$SETUPTOOLS_VERSION
