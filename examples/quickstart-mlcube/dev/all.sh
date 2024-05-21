#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Cleanup & prepare mlcube
./dev/setup.sh

# Start server
./dev/server.sh &
sleep 2 # Sleep for 2s to give the server enough time to start

for i in `seq 0 1`; do
    ./dev/client.sh $i &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
