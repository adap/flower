#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

echo "Starting server"
python3 server.py --pool-size=5 --num-rounds=30 --num-clients-per-round=5 --centralised-eval &
sleep 30  # Sleep for 30s to give the server enough time to start

for i in `seq 0 4`; do
    echo "Starting client $i"
    python3 client.py --partition-id=$i --num-partitions=5 --partitioner-type=exponential &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
