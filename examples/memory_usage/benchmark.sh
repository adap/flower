#!/bin/bash
set -e

NUM_CLIENTS=200

# Clean up
rm -rf results
mkdir results


# Benchmark
python -m mprof run --include-children -o results/memory_usage_server.dat server.py &
sleep 2

for (( i=0; i < $NUM_CLIENTS; i++ ))
do
    python -m mprof run --include-children -o results/memory_usage_client_$i.dat client.py &
    sleep 0.1
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

wait

