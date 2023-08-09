#!/bin/bash

echo "Starting server"
docker run --rm -it --platform linux/amd64 --detach --name server flwr_waterlily waterlily.server
sleep 3  # Sleep for 3s to give the server enough time to start


for i in `seq 0 1`; do
    echo "Starting client $i"
    docker run --rm -it --platform linux/amd64 flwr_waterlily waterlily.client server
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait

docker rm -f flwr_waterlily*
