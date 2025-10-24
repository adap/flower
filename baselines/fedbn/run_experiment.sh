#!/bin/bash
for algorithm in "FedAvg" "FedBN" ; # Defining the Algorithm types fraction
do
  for i in $(seq 1 5) # Defining five repetitions for each experiment type
    do
        flwr run . --run-config "algorithm-name='${algorithm}' num-server-rounds=100"
    done
done