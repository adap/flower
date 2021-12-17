#!/bin/bash
mode=${1:-"fedbn"}

echo "Start Trainign with $mode"

python server.py &
sleep 5 # Sleep for 2s to give the server enough time to start

for i in 'MNIST' 'SVHN' 'USPS' 'SynthDigits' 'MNIST-M'; do
    touch "${i}_"$mode"_results.json"
    sleep 2 
done

for i in 'MNIST' 'SVHN' 'USPS' 'SynthDigits' 'MNIST-M' ; do
    echo "Starting client $i"
    python client.py --partition=${i} --mode="$mode" &
    sleep 2 &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
