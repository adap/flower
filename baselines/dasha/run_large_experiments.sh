#!/bin/bash
flwr run . --run-config conf/large_stochastic_dasha.toml gpu-simulation
flwr run . --run-config conf/large_stochastic_marina.toml gpu-simulation

