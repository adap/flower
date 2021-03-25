#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

NUM_CLIENTS=10
for ((i=0; i<$NUM_CLIENTS; i++)); do
    docker build -f ./docker/client.Dockerfile --build-arg index=$i -t flower_client_$i .
done
