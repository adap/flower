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

set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../../../

docker run -d --rm --network flower --name logserver flower:latest \
  python3.7 -m flwr_experimental.logserver

docker run -d --rm --network flower --name server flower:latest \
  python3.7 -m flwr_example.tf_fashion_mnist.server \
  --rounds=10 \
  --sample_fraction=0.5 \
  --min_sample_size=4 \
  --min_num_clients=4 \
  --log_host=logserver:8081

docker run -d --rm --network flower --name client_0 flower:latest \
  python3.7 -m flwr_example.tf_fashion_mnist.client --cid=0 --partition=0 --clients=4 --server_address=server:8080 --log_host=logserver:8081

docker run -d --rm --network flower --name client_1 flower:latest \
  python3.7 -m flwr_example.tf_fashion_mnist.client --cid=1 --partition=1 --clients=4 --server_address=server:8080 --log_host=logserver:8081

docker run -d --rm --network flower --name client_2 flower:latest \
  python3.7 -m flwr_example.tf_fashion_mnist.client --cid=2 --partition=2 --clients=4 --server_address=server:8080 --log_host=logserver:8081

docker run -d --rm --network flower --name client_3 flower:latest \
  python3.7 -m flwr_example.tf_fashion_mnist.client --cid=3 --partition=3 --clients=4 --server_address=server:8080 --log_host=logserver:8081

docker exec logserver tail -f flower_logs/flower.log
