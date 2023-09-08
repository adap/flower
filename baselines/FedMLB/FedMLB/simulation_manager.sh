#!/usr/bin/bash
rounds_per_run=20
total_rounds=1000
iterations=$total_rounds/$rounds_per_run

for (( i=1; i <= iterations; ++i ))
do
    echo "$i"
    python -m FedMLB.main num_rounds=$rounds_per_run dataset_config.dataset="tiny-imagenet" algorithm="FedAvg" total_clients=500 clients_per_round=10
done
