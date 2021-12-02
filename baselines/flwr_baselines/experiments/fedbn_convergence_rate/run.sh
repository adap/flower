#!/bin/bash

python server.py &
sleep 5 # Sleep for 2s to give the server enough time to start

# 'MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST-M'

for i in 'MNIST' 'SVHN' 'USPS' 'SynthDigits' 'MNIST-M' ; do
    echo "Starting client $i"
    python client.py --partition=${i} &
    sleep 2 &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
