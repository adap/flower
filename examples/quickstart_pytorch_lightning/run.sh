#!/bin/bash

# Group all processes in wandb
export WANDB_RUN_GROUP=`python -c "import wandb; print(wandb.util.generate_id())"`

python server.py &
sleep 3 # Sleep for 3s to give the server enough time to start

for i in `seq 0 1`; do
    echo "Starting client $i"
    python client.py &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
