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

# This script is regularly used to create an AWS EC2 AMI which has most dependencies needed
# for the flower benchmarks to execute. The AWS EC2 AMI might not always up to date but we will
# try to keep it so.

# This script was last time used on 2020-04-13 to create an AMI with ID ami-0396b4e13e2f040cc
# The AMI name is flower-2020-04-20

# Prepare machine dependencies
sudo apt update
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl git
sudo apt install python3.7
sudo apt install python3-pip

# Install project dependencies
python3.7 -m pip install numpy==1.18.1 grpcio==1.27.2 google==2.0.3 tensorflow-cpu==2.1.0 boto3==1.12.36 boto3_type_annotations==0.3.1 paramiko==2.7.1

# Preload datasets
python3.7 -c "import tensorflow as tf; tf.keras.datasets.mnist.load_data()"
python3.7 -c "import tensorflow as tf; tf.keras.datasets.fashion_mnist.load_data()"
python3.7 -c "import tensorflow as tf; tf.keras.datasets.cifar10.load_data()"
python3.7 -c "import tensorflow as tf; tf.keras.datasets.cifar100.load_data()"
