#!/bin/bash
flwr run . --run-config conf/large_stochastic_dasha.toml
flwr run . --run-config conf/large_stochastic_marina.toml

