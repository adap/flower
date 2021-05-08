#!/bin/bash

python server.py & 
sleep 2 # Sleep for 2s to give the server enough time to start
python client.py &
python client.py &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
sleep 86400
