#!/bin/bash
for sf in 0.0 0.5 0.9 # Defining the stragglers fraction
do
  for i in $(seq 1 5) # Defining five repetitions for each experiment type
    do
        flwr run . femnist-federation --run-config conf/femnist/fedavg_sf_$sf.toml
        flwr run . femnist-federation --run-config conf/femnist/fedprox_mu_0.0_sf_$sf.toml
        flwr run . femnist-federation --run-config conf/femnist/fedprox_mu_2.0_sf_$sf.toml
    done
done