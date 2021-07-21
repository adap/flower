#!/bin/bash

set -e

pushd client_1/mlcube
poetry run mlcube_docker configure --mlcube=. --platform=platforms/docker.yaml
popd

python server.py &
sleep 2 # Sleep for 2s to give the server enough time to start
(cd client_1 && python client.py) &
(cd client_2 && python client.py) &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
