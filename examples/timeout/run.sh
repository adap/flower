#!/bin/bash

echo "Starting server"
python server.py &
sleep 3  # Sleep for 3s to give the server enough time to start

python client_fast.py &
python client_fast.py &
python client_slow.py &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
