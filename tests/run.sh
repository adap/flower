#!/bin/bash
set -e

poetry run python server.py &
sleep 2 # Sleep for 2s to give the server enough time to start

for i in `seq 0 4`; do
    echo "Starting client $i"
    poetry run python client.py --partition=${i} &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM KILL
# Wait for all background processes to complete
wait
