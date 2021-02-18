#!/bin/bash

python server.py & 
sleep 2 # Sleep for 2s to give the server enough time to start
python client.py --partition=0 &
python client.py --partition=1 &
python client.py --partition=2 &
python client.py --partition=3 &
python client.py --partition=4 &
python client.py --partition=5 &
python client.py --partition=6 &
python client.py --partition=7 &
python client.py --partition=8 &
python client.py --partition=9 &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
sleep 86400
