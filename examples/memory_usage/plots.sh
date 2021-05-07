#!/bin/bash
set -e

NUM_CLIENTS=200

# Plots
python -m mprof plot results/memory_usage_server.dat -o results/memory_usage_server.jpg

for (( i=0; i < $NUM_CLIENTS; i++ ))
do
    python -m mprof plot results/memory_usage_client_$i.dat -o results/memory_usage_client_$i.jpg
done

