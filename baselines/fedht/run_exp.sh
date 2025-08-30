#!/bin/bash

flwr run . --run-config "agg='fedavg' num_keep=500 num_local_epochs=1 learning_rate=0.00001"
flwr run . --run-config "agg='fedht' num_keep=500 num_local_epochs=1 learning_rate=0.00001"
flwr run . --run-config "agg='fedht' iterht=true num_keep=500 num_local_epochs=1 learning_rate=0.00001"

flwr run . --run-config "agg='fedavg' num_keep=500 num_local_epochs=5 learning_rate=0.00001"
flwr run . --run-config "agg='fedht' num_keep=500 num_local_epochs=5 learning_rate=0.00001"
flwr run . --run-config "agg='fedht' iterht=true num_keep=500 num_local_epochs=5 learning_rate=0.00001"

flwr run . --run-config "agg='fedavg' num_keep=500 num_local_epochs=1 learning_rate=0.00001 seed=1"
flwr run . --run-config "agg='fedavg' num_keep=500 num_local_epochs=1 learning_rate=0.00001 seed=2"
flwr run . --run-config "agg='fedavg' num_keep=500 num_local_epochs=1 learning_rate=0.00001 seed=3"
flwr run . --run-config "agg='fedavg' num_keep=500 num_local_epochs=1 learning_rate=0.00001 seed=4"
flwr run . --run-config "agg='fedavg' num_keep=500 num_local_epochs=1 learning_rate=0.00001 seed=5"

flwr run . --run-config "agg='fedht' num_keep=700 num_local_epochs=1 learning_rate=0.00001 seed=1"
flwr run . --run-config "agg='fedht' num_keep=700 num_local_epochs=1 learning_rate=0.00001 seed=2"
flwr run . --run-config "agg='fedht' num_keep=700 num_local_epochs=1 learning_rate=0.00001 seed=3"
flwr run . --run-config "agg='fedht' num_keep=700 num_local_epochs=1 learning_rate=0.00001 seed=4"
flwr run . --run-config "agg='fedht' num_keep=700 num_local_epochs=1 learning_rate=0.00001 seed=5"

flwr run . --run-config "agg='fedht' num_keep=500 num_local_epochs=1 learning_rate=0.00001 seed=1"
flwr run . --run-config "agg='fedht' num_keep=500 num_local_epochs=1 learning_rate=0.00001 seed=2"
flwr run . --run-config "agg='fedht' num_keep=500 num_local_epochs=1 learning_rate=0.00001 seed=3"
flwr run . --run-config "agg='fedht' num_keep=500 num_local_epochs=1 learning_rate=0.00001 seed=4"
flwr run . --run-config "agg='fedht' num_keep=500 num_local_epochs=1 learning_rate=0.00001 seed=5"

flwr run . --run-config "agg='fedht' num_keep=250 num_local_epochs=1 learning_rate=0.00001 seed=1"
flwr run . --run-config "agg='fedht' num_keep=250 num_local_epochs=1 learning_rate=0.00001 seed=2"
flwr run . --run-config "agg='fedht' num_keep=250 num_local_epochs=1 learning_rate=0.00001 seed=3"
flwr run . --run-config "agg='fedht' num_keep=250 num_local_epochs=1 learning_rate=0.00001 seed=4"
flwr run . --run-config "agg='fedht' num_keep=250 num_local_epochs=1 learning_rate=0.00001 seed=5"

flwr run . --run-config "agg='fedht' num_keep=100 num_local_epochs=1 learning_rate=0.00001 seed=1"
flwr run . --run-config "agg='fedht' num_keep=100 num_local_epochs=1 learning_rate=0.00001 seed=2"
flwr run . --run-config "agg='fedht' num_keep=100 num_local_epochs=1 learning_rate=0.00001 seed=3"
flwr run . --run-config "agg='fedht' num_keep=100 num_local_epochs=1 learning_rate=0.00001 seed=4"
flwr run . --run-config "agg='fedht' num_keep=100 num_local_epochs=1 learning_rate=0.00001 seed=5"

flwr run . --run-config "agg='fedht' num_keep=50 num_local_epochs=1 learning_rate=0.00001 seed=1"
flwr run . --run-config "agg='fedht' num_keep=50 num_local_epochs=1 learning_rate=0.00001 seed=2"
flwr run . --run-config "agg='fedht' num_keep=50 num_local_epochs=1 learning_rate=0.00001 seed=3"
flwr run . --run-config "agg='fedht' num_keep=50 num_local_epochs=1 learning_rate=0.00001 seed=4"
flwr run . --run-config "agg='fedht' num_keep=50 num_local_epochs=1 learning_rate=0.00001 seed=5"

flwr run . --run-config "agg='fedht' num_keep=25 num_local_epochs=1 learning_rate=0.00001 seed=1"
flwr run . --run-config "agg='fedht' num_keep=25 num_local_epochs=1 learning_rate=0.00001 seed=2"
flwr run . --run-config "agg='fedht' num_keep=25 num_local_epochs=1 learning_rate=0.00001 seed=3"
flwr run . --run-config "agg='fedht' num_keep=25 num_local_epochs=1 learning_rate=0.00001 seed=4"
flwr run . --run-config "agg='fedht' num_keep=25 num_local_epochs=1 learning_rate=0.00001 seed=5"

flwr run . --run-config "agg='fedht' num_keep=25 num_local_epochs=1 learning_rate=0.00001 seed=1"
flwr run . --run-config "agg='fedht' num_keep=25 num_local_epochs=1 learning_rate=0.00001 seed=2"
flwr run . --run-config "agg='fedht' num_keep=25 num_local_epochs=1 learning_rate=0.00001 seed=3"
flwr run . --run-config "agg='fedht' num_keep=25 num_local_epochs=1 learning_rate=0.00001 seed=4"
flwr run . --run-config "agg='fedht' num_keep=25 num_local_epochs=1 learning_rate=0.00001 seed=5"