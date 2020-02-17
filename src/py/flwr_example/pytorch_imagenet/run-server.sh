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

IMAGENET_PATH="~/Downloads/imagenet-object-localization-challenge/"

# Start a Flower server
python -m flwr_example.pytorch_imagenet.server \
  --rounds=100 \
  --sample_fraction=0.25 \
  --min_sample_size=10 \
  --min_num_clients=30 \
  --data_path=$IMAGENET_PATH
