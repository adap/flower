#!/bin/bash
rm -f logger.log
echo "Starting server"
nohup python server.py  >>logger.log &
sleep 3  # Sleep for 3s to give the server enough time to start

for i in `seq 0 2`; do
    echo "Starting client $i"
    nohup python client.py --part_idx $i  >>logger.log &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
# SOS killall python