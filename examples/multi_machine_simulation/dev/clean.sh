#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

docker rm -f flower_server || true

NUM_CLIENTS=10
for ((i=0; i<$NUM_CLIENTS; i++)); do
    docker rm -f flower_client_$i || true
done
