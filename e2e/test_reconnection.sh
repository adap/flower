#!/bin/bash
set -e

case "$1" in
  rest)
    rest_arg="--rest"
    server_app_address="http://localhost:9091"
    server_address="http://localhost:9093"
    db_arg="--database :flwr-in-memory-state:"
    ;;
  sqlite)
    rest_arg=""
    server_address="127.0.0.1:9092"
    server_app_address="127.0.0.1:9091"
    db_arg="--database $(date +%s).db"
    ;;
  *)
    rest_arg=""
    server_address="127.0.0.1:9092"
    server_app_address="127.0.0.1:9091"
    db_arg="--database :flwr-in-memory-state:"
    ;;
esac

dir_arg="--dir ./.."

timeout 2m flower-superlink --insecure $db_arg $rest_arg &
sl_pid=$!
echo "Starting SuperLink"
sleep 3

timeout 2m flower-client-app client:app --insecure $rest_arg --server $server_address &
cl1_pid=$!
echo "Starting first client"
sleep 3

timeout 2m flower-client-app client:app --insecure $rest_arg --server $server_address &
cl2_pid=$!
echo "Starting second client"
sleep 3

# Kill superlink, this should send the clients into their retry loops
kill $sl_pid
echo "Killing Superlink"
sleep 3

# Restart superlink, the clients should now be able to reconnect to it
timeout 2m flower-superlink --insecure $db_arg $rest_arg &
sl_pid=$!
echo "Restarting Superlink"
sleep 20

# Kill first client, this should send a DeleteNode message to the Superlink
kill $cl1_pid
echo "Killing first client"
sleep 3

# Starting new client, this is so we have enough clients to start the server-app
timeout 2m flower-client-app client:app --insecure $rest_arg --server $server_address &
cl1_pid=$!
echo "Starting new client"
sleep 5

# We start the server-app to begining the training
timeout 2m flower-server-app server:app --insecure $dir_arg $rest_arg --server $server_app_address &
pid=$!
echo "Starting server-app to start training"

# Kill first client as soon as the training starts,
# the server-app should just receive a failure in this case and continue the rounds
# when enough clients are connected
kill $cl1_pid
echo "Killing first client"
sleep 1

# Restart first client so enough clients are connected to continue the FL rounds
timeout 2m flower-client-app client:app --insecure $rest_arg --server $server_address &
cl1_pid=$!
echo "Starting new client"

wait $pid
res=$?

if [[ "$res" = "0" ]];
  then echo "Training worked correctly"; kill $cl1_pid; kill $cl2_pid; kill $sl_pid;
  else echo "Training had an issue" && exit 1;
fi

