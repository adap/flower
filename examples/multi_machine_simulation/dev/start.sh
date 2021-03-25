#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

./dev/build_server.sh
./dev/build_clients.sh

# Create network if not exists so the containers can reach each other by name
# This is important as the clients will use the server name as hostname
docker network create --driver bridge flower_network || true

docker run \
    --rm \
    --detach \
    --name flower_server \
    --network=flower_network \
    flower_server

# Loop from 0 to i<$NUM_CLIENTS and start the
# respective flower_client image
NUM_CLIENTS=10
for ((i=0; i<$NUM_CLIENTS; i++)); do
    docker run \
        --rm \
        --detach \
        --name flower_client_$i \
        --network=flower_network \
        flower_client_$i flower_server:8080
done

docker ps -a
docker logs -f flower_server
