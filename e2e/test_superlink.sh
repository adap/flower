#!/bin/bash
set -e

case "$1" in
  e2e-bare-https | e2e-bare-auth)
    ./generate.sh
    server_arg="--ssl-ca-certfile certificates/ca.crt --ssl-certfile certificates/server.pem --ssl-keyfile certificates/server.key"
    client_arg="--root-certificates certificates/ca.crt"
    server_dir="./"
    ;;
  *)
    server_arg="--insecure"
    client_arg="--insecure"
    server_dir="./"
    ;;
esac

case "$2" in
  rest)
    rest_arg_superlink="--fleet-api-type rest"
    rest_arg_supernode="--rest"
    server_address="http://localhost:9095"
    server_app_address="127.0.0.1:9091"
    db_arg="--database :flwr-in-memory-state:"
    server_auth=""
    client_auth_1=""
    client_auth_2=""
    ;;
  sqlite)
    rest_arg_superlink=""
    rest_arg_supernode=""
    server_address="127.0.0.1:9092"
    server_app_address="127.0.0.1:9091"
    db_arg="--database $(date +%s).db"
    server_auth=""
    client_auth_1=""
    client_auth_2=""
    ;;
  client-auth)
    rest_arg_superlink=""
    rest_arg_supernode=""
    server_address="127.0.0.1:9092"
    server_app_address="127.0.0.1:9091"
    db_arg="--database :flwr-in-memory-state:"
    server_auth="--auth-list-public-keys keys/client_public_keys.csv --auth-superlink-private-key keys/server_credentials --auth-superlink-public-key keys/server_credentials.pub"
    client_auth_1="--auth-supernode-private-key keys/client_credentials_1 --auth-supernode-public-key keys/client_credentials_1.pub"
    client_auth_2="--auth-supernode-private-key keys/client_credentials_2 --auth-supernode-public-key keys/client_credentials_2.pub"
    ;;
  *)
    rest_arg_superlink=""
    rest_arg_supernode=""
    server_address="127.0.0.1:9092"
    server_app_address="127.0.0.1:9091"
    db_arg="--database :flwr-in-memory-state:"
    server_auth=""
    client_auth_1=""
    client_auth_2=""
    ;;
esac

# Install Flower app
pip install -e . --no-deps

# Remove any duplicates
sed -i '/^\[tool\.flwr\.federations\.e2e\]/,/^$/d' pyproject.toml

# Check if the first argument is 'insecure'
if [ "$server_arg" = "--insecure" ]; then
  # If $server_arg is '--insecure', append the first line
  echo -e $"\n[tool.flwr.federations.e2e]\naddress = \"127.0.0.1:9093\"\ninsecure = true" >> pyproject.toml
else
  # Otherwise, append the second line
  echo -e $"\n[tool.flwr.federations.e2e]\naddress = \"127.0.0.1:9093\"\nroot-certificates = \"certificates/ca.crt\"" >> pyproject.toml
fi

timeout 5m flower-superlink $server_arg $db_arg $rest_arg_superlink $server_auth \
  2>&1 | tee flwr_output.log &
sl_pid=$(pgrep -f "flower-superlink")
sleep 3

timeout 5m flower-supernode $client_arg $rest_arg_supernode \
  --superlink $server_address $client_auth_1 \
  --clientappio-api-address "localhost:9094" \
  --max-retries 0 &
cl1_pid=$!
sleep 3

timeout 5m flower-supernode $client_arg $rest_arg_supernode \
  --superlink $server_address $client_auth_2 \
  --clientappio-api-address "localhost:9096" \
  --max-retries 0 &
cl2_pid=$!
sleep 3

timeout 1m flwr run "." e2e

# Initialize a flag to track if training is successful
found_success=false
timeout=240  # Timeout after 240 seconds
elapsed=0

# Check for "Success" in a loop with a timeout
while [ "$found_success" = false ] && [ $elapsed -lt $timeout ]; do
    if grep -q "Run finished" flwr_output.log; then
        echo "Training worked correctly!"
        found_success=true
        kill $cl1_pid; kill $cl2_pid;
        sleep 1; kill $sl_pid;
    else
        echo "Waiting for training ... ($elapsed seconds elapsed)"
    fi
    # Sleep for a short period and increment the elapsed time
    sleep 2
    elapsed=$((elapsed + 2))
done

if [ "$found_success" = false ]; then
    echo "Training had an issue and timed out."
    kill $cl1_pid; kill $cl2_pid;
    kill $sl_pid;
fi
