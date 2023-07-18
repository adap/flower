#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# Download the CIFAR-10 dataset
python -c "from torchvision.datasets import CIFAR10; CIFAR10('./data', download=True)"

echo "Starting server"
python server.py &
sleep 3  # Sleep for 3s to give the server enough time to start

for i in `seq 0 1`; do
    echo "Starting client $i"
    python client.py &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
