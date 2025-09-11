#!/bin/bash
for step in 0.25 0.5 1.0 # Defining the step size
do
    flwr run . --run-config conf/small_dasha_$step.toml
    flwr run . --run-config conf/small_marina_$step.toml
done