#!/bin/bash

# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
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

set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

BROWN='\033[0;33m'
NC='\033[0m' # No Color

# Define default values first
BUILD_STAGE="${BUILD_STAGE:=build}"
TAG="${TAG:=`cat ../../pyproject.toml | grep "^version = " | awk '{print $3}' | tr -d '"'`}"
PYTHON_VERSION="${PYTHON_VERSION:=3.9.17}"
POETRY_VERSION="${POETRY_VERSION:=1.5.1}"

echo -e "${BROWN}\nUsing:"
echo -e "BUILD_STAGE: $BUILD_STAGE"
echo -e "TAG: $TAG"
echo -e "PYTHON_VERSION: $PYTHON_VERSION"
echo -e "POETRY_VERSION: $POETRY_VERSION"
echo -e "${NC}"

echo -e "${BROWN}\nBuilding image base with tag $TAG${NC}"
docker build \
    --target $BUILD_STAGE \
    -f 10_base.Dockerfile \
    --build-arg PYTHON_VERSION=$PYTHON_VERSION \
    --build-arg POETRY_VERSION=$POETRY_VERSION \
    -t flwr/base:$TAG \
    -t flwr/base:latest \
    .

echo -e "${BROWN}\nBuilding image server with tag $TAG${NC}"
docker build \
    --target $BUILD_STAGE \
    -f 20_server.Dockerfile \
    --build-arg BASE_VERSION=$TAG \
    -t flwr/server:$TAG \
    -t flwr/server:latest \
    .
