#!/bin/bash

# Copyright 2020 Adap GmbH. All Rights Reserved.
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

# This script can be used to create an AWS EC2 AMI which contains the dependencies required
# to execute Flower PyTorch based baselines. The AWS EC2 AMI might not always reflect all
# dependencies listed in `pyproject.toml`, but it should at least have most of them.

# Prepare machine dependencies
sudo apt update
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev \
    liblzma-dev python-openssl git
sudo apt install -y python3.7 python3-pip

# Install project dependencies
python3.7 -m pip install -U pip==22.2 setuptools==63.2.0
python3.7 -m pip install -U numpy==1.18.1 grpcio==1.27.2 google==2.0.3 protobuf==3.12.1 \
    boto3==1.12.36 boto3_type_annotations==0.3.1 paramiko==2.7.1 docker==4.2.0 matplotlib==3.2.1 \
    tqdm==4.48.2 torch==1.6.0 torchvision==0.7.0

# Preload datasets
python3.7 -m flwr_experimental.baseline.dataset.pytorch_cifar_partitioned
