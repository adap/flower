#!/bin/bash
set -e

# Set connectivity parameters
case "$1" in
    secure)
      ./generate.sh
      server_arg='--ssl-ca-certfile ../certificates/ca.crt
                  --ssl-certfile    ../certificates/server.pem
                  --ssl-keyfile     ../certificates/server.key'
      client_arg='--root-certificates ../certificates/ca.crt'
      # For $executor_config, note special ordering of single- and double-quotes
      executor_config='root-certificates="../certificates/ca.crt"'
      ;;
    insecure)
      server_arg='--insecure'
      client_arg=$server_arg
      executor_config=''
    ;;
esac

# Set authentication parameters
case "$2" in
    client-auth)
      server_auth='--auth-list-public-keys      ../keys/client_public_keys.csv'
      client_auth_1='--auth-supernode-private-key ../keys/client_credentials_1 
                     --auth-supernode-public-key  ../keys/client_credentials_1.pub'
      client_auth_2='--auth-supernode-private-key ../keys/client_credentials_2 
                     --auth-supernode-public-key  ../keys/client_credentials_2.pub'
      server_address='127.0.0.1:9092'
      ;;
    *)
    server_auth=''
    client_auth_1=''
    client_auth_2=''
    server_address='127.0.0.1:9092'
    ;;
esac

# Set engine
case "$3" in
    deployment-engine)
      executor_arg="--executor flwr.superexec.deployment:executor"
      ;;
    simulation-engine)
      executor_arg="--executor flwr.superexec.simulation:executor"
      ;;
esac


# Create and install Flower app
flwr new e2e-tmp-test --framework numpy --username flwrlabs
cd e2e-tmp-test
# Remove flwr dependency from `pyproject.toml`. Seems necessary so that it does
# not override the wheel dependency
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS (Darwin) system
    sed -i '' '/flwr\[simulation\]/d' pyproject.toml
else
    # Non-macOS system (Linux)
    sed -i '/flwr\[simulation\]/d' pyproject.toml
fi
pip install -e . --no-deps

# Check if the first argument is 'insecure'
if [ "$1" == "insecure" ]; then
  # If $1 is 'insecure', append the first line
  echo -e $"\n[tool.flwr.federations.e2e]\naddress = \"127.0.0.1:9093\"\ninsecure = true" >> pyproject.toml
else
  # Otherwise, append the second line
  echo -e $"\n[tool.flwr.federations.e2e]\naddress = \"127.0.0.1:9093\"\nroot-certificates = \"../certificates/ca.crt\"" >> pyproject.toml
fi

if [ "$3" = "simulation-engine" ]; then
  echo -e $"options.num-supernodes = 10" >> pyproject.toml
fi

# Combine the arguments into a single command for flower-superlink
combined_args="$server_arg $server_auth $exec_api_arg $executor_arg"

timeout 2m flower-superlink $combined_args --executor-config "$executor_config" 2>&1 | tee flwr_output.log &
sl_pid=$(pgrep -f "flower-superlink")
sleep 2

if [ "$3" = "deployment-engine" ]; then
  timeout 2m flower-supernode $client_arg \
      --superlink $server_address $client_auth_1 \
      --clientappio-api-address localhost:9094 \
      --node-config "partition-id=0 num-partitions=2" --max-retries 0 &
  cl1_pid=$!
  sleep 2

  timeout 2m flower-supernode $client_arg \
      --superlink $server_address $client_auth_2 \
      --clientappio-api-address localhost:9095 \
      --node-config "partition-id=1 num-partitions=2" --max-retries 0 &
  cl2_pid=$!
  sleep 2
fi

timeout 1m flwr run --run-config num-server-rounds=1 ../e2e-tmp-test e2e

# Initialize a flag to track if training is successful
found_success=false
timeout=120  # Timeout after 120 seconds
elapsed=0

# Check for "Success" in a loop with a timeout
while [ "$found_success" = false ] && [ $elapsed -lt $timeout ]; do
    if grep -q "Run finished" flwr_output.log; then
        echo "Training worked correctly!"
        found_success=true
        if [ "$3" = "deployment-engine" ]; then
          kill $cl1_pid; kill $cl2_pid;
        fi
        sleep 1; kill $sl_pid;
        exit 0;
    else
        echo "Waiting for training ... ($elapsed seconds elapsed)"
    fi
    # Sleep for a short period and increment the elapsed time
    sleep 2
    elapsed=$((elapsed + 2))
done

if [ "$found_success" = false ]; then
    echo "Training had an issue and timed out."
    if [ "$3" = "deployment-engine" ]; then
      kill $cl1_pid; kill $cl2_pid;
    fi
    kill $sl_pid;
    exit 1;
fi
