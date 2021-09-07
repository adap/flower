#!/bin/bash
set -e

docker build -t flower_federated_learning_simulation_pytorch . && \
docker run --ipc=host -it --rm flower_federated_learning_simulation_pytorch
