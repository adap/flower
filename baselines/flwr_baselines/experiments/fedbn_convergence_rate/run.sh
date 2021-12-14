#!/bin/bash

echo "Start Trainign with FedAvg"

python server.py &
sleep 5 # Sleep for 2s to give the server enough time to start

for i in 'MNIST' 'SVHN' 'USPS' 'SynthDigits' 'MNIST-M'; do
    touch "${i}_fedavg_results.json"
    sleep 2 
done

for i in 'MNIST' 'SVHN' 'USPS' 'SynthDigits' 'MNIST-M' ; do
    echo "Starting client $i"
    python client.py --partition=${i} --mode="fedavg" &
    sleep 2 &
done


# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait

echo "Start Trainign with FedBN"

for i in 'MNIST' 'SVHN' 'USPS' 'SynthDigits' 'MNIST-M'; do
    touch "${i}_fedbn_results.json"
    sleep 2 
done

for i in 'MNIST' 'SVHN' 'USPS' 'SynthDigits' 'MNIST-M' ; do
    echo "Starting client $i"
    python client.py --partition=${i} --mode="fedbn" &
    sleep 2 &
done