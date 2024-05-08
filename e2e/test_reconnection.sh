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
sleep 3

timeout 2m flower-client-app client:app --insecure $rest_arg --server $server_address &
cl1_pid=$!
sleep 3

timeout 2m flower-client-app client:app --insecure $rest_arg --server $server_address &
cl2_pid=$!
sleep 3

# Kill superlink
kill $sl_pid
sleep 3

# Restart superlink
timeout 2m flower-superlink --insecure $db_arg $rest_arg &
sl_pid=$!
sleep 20

# Kill first client
kill $cl1_pid
sleep 3

# Restart first client
timeout 2m flower-client-app client:app --insecure $rest_arg --server $server_address &
cl1_pid=$!
sleep 5

timeout 2m flower-server-app server:app --insecure $dir_arg $rest_arg --server $server_app_address &
pid=$!

# Kill first client
kill $cl1_pid
sleep 1

# Restart first client
timeout 2m flower-client-app client:app --insecure $rest_arg --server $server_address &
cl1_pid=$!

wait $pid
res=$?

if [[ "$res" = "0" ]];
  then echo "Training worked correctly"; kill $cl1_pid; kill $cl2_pid; kill $sl_pid;
  else echo "Training had an issue" && exit 1;
fi

