#!/bin/bash
set -e

docker build -t flower_federated_learning_simulation . && \
docker run -it --rm flower_federated_learning_simulation
