#!/usr/bin/env bash

# Create and install Flower app
flwr new e2e-tmp-test --framework numpy --username flwrlabs
cd e2e-tmp-test

# Modify the config file
echo -e $"\n[tool.flwr.federations.e2e]\naddress = \"127.0.0.1:9093\"\ninsecure = true" >> pyproject.toml

# Start Flower processes in the background
flower-superlink --insecure 2>&1 | tee flwr_output.log &
sleep 2

flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address localhost:9094 \
    --max-retries 0 &
cl1_pid=$!
sleep 2

flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address localhost:9095 \
    --max-retries 0 &
sleep 2

flwr run --run-config num-server-rounds=1 . e2e

# Trap to clean up on exit
cleanup() {
    echo "Stopping Flower processes..."
    taskkill //F //FI "IMAGENAME eq flower*" //T
}
trap cleanup EXIT

# Initialize a flag to track if training is successful
found_success=false
timeout=120  # Timeout after 120 seconds
elapsed=0

# Check for "Run finished" in a loop with a timeout
while [ "$found_success" = false ] && [ $elapsed -lt $timeout ]; do
    if grep -q "Run finished" flwr_output.log; then
        echo "Training worked correctly!"
        found_success=true
        exit 0;
    else
        echo "Waiting for training ... ($elapsed seconds elapsed)"
    fi
    # Sleep for a short period and increment the elapsed time
    sleep 2
    elapsed=$((elapsed + 2))
done

if [ "$found_success" = false ]; then
    echo "Training did not finish within timeout."
    exit 1;
fi
