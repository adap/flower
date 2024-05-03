#!/bin/bash
set -e

case "$1" in
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

timeout 2m flower-superlink $db_arg $rest_arg &
sl_pid=$!
sleep 3

timeout 2m flower-client-app client:app $rest_arg --server $server_address &
cl1_pid=$!
sleep 3

timeout 2m flower-client-app client:app $rest_arg --server $server_address &
cl2_pid=$!
sleep 3

timeout 2m flower-server-app server:app $rest_arg --server $server_address &
pid=$!

wait $pid
res=$?

if [[ "$res" = "0" ]];
  then echo "Training worked correctly"; kill $cl1_pid; kill $cl2_pid; kill $sl_pid;
  else echo "Training had an issue" && exit 1;
fi

