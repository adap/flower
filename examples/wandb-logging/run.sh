#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# Download the MNIST dataset
python -c "from torchvision.datasets import MNIST; MNIST('./data', download=True)"

echo "Starting server"
python server.py &
sleep 10 # Sleep for 10s to give the server enough time to start

for i in `seq 0 4`; do
    echo "Starting client $i"
    python client.py --cid $i &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
