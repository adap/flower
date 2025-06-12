#!/bin/bash
for sf in 0.0 0.5 0.9 # Defining the stragglers fraction
do
  for i in $(seq 1 5) # Defining five repetitions for each experiment type
    do
        flwr run . --run-config conf/mnist/fedavg_sf_$sf.toml
        flwr run . --run-config conf/mnist/fedprox_mu_0.0_sf_$sf.toml
        flwr run . --run-config conf/mnist/fedprox_mu_2.0_sf_$sf.toml
    done
done