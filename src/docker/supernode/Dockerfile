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

ARG BASE_REPOSITORY=flwr/base
ARG PYTHON_VERSION=3.11
ARG UBUNTU_VERSION=ubuntu22.04
FROM $BASE_REPOSITORY:py${PYTHON_VERSION}-${UBUNTU_VERSION}

ARG FLWR_PACKAGE=flwr
ARG FLWR_VERSION
RUN python -m pip install -U --no-cache-dir \
    ${FLWR_PACKAGE}==${FLWR_VERSION} && \
    pyenv rehash

WORKDIR /app
ENTRYPOINT ["flower-client-app"]
