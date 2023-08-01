#!/bin/bash
mode=${1:-"fedavg"}

echo "Start Training with $mode"
mkdir -p results

python3 server.py &
sleep 5 # Sleep for 5s to give the server enough time to start

for i in 'MNIST' 'SVHN' 'USPS' 'SynthDigits' 'MNIST-M'; do
    touch "results/${i}_"$mode"_results.json"
    sleep 5 
done

for i in 'MNIST' 'SVHN' 'USPS' 'SynthDigits' 'MNIST-M' ; do
    echo "Starting client $i"
    python3 client.py --partition=${i} --mode="$mode" &
    sleep 5 &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
