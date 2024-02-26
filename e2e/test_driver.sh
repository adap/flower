#!/bin/bash
set -e

case "$1" in
  bare-https)
    ./generate.sh
    server_arg="--certificates certificates/ca.crt certificates/server.pem certificates/server.key"
    client_arg="--root-certificates certificates/ca.crt"
    ;;
  *)
    server_arg="--insecure"
    client_arg="--insecure"
    ;;
esac

case "$2" in
  rest)
    rest_arg="--rest"
    server_address="http://localhost:9093"
    db_arg="--database :flwr-in-memory-state:"
    ;;
  sqlite)
    rest_arg=""
    server_address="127.0.0.1:9092"
    db_arg="--database $(date +%s).db"
    ;;
  *)
    rest_arg=""
    server_address="127.0.0.1:9092"
    db_arg="--database :flwr-in-memory-state:"
    ;;
esac

timeout 2m flower-superlink $server_arg $db_arg $rest_arg &
sl_pid=$!
sleep 3

timeout 2m flower-client-app client:app $client_arg $rest_arg --server $server_address &
cl1_pid=$!
sleep 3

timeout 2m flower-client-app client:app $client_arg $rest_arg --server $server_address &
cl2_pid=$!
sleep 3

timeout 2m python driver.py &
pid=$!

wait $pid
res=$?

if [[ "$res" = "0" ]];
  then echo "Training worked correctly"; kill $cl1_pid; kill $cl2_pid; kill $sl_pid;
  else echo "Training had an issue" && exit 1;
fi

