#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# Download the CIFAR-10 dataset
python -c "from torchvision.datasets import CIFAR10; CIFAR10('./data', download=True)"

echo "Starting server"
python server.py &
sleep 3  # Sleep for 3s to give the server enough time to start
IMG_ROOT='/datasets/FedScale/openImg/'
CID_CSV_ROOT='/datasets/FedScale/openImg/client_data_mapping/clean_ids'

for i in `seq 0 1`; do
    echo "Starting client $i"
    python worker.py -- &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
