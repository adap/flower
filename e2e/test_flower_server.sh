#!/bin/bash
set -e

case $1 in
  "rere")
    server_args="--grpc-rere --grpc-rere-fleet-api-address"
    client_args="grpc-rere"
    ;;

  "rest")
    server_args="--rest --rest-fleet-api-address"
    client_args="rest"
    ;;

  *)
    server_args="--grpc-bidi --grpc-bidi-fleet-api-address"
    client_args="grpc-bidi"
    ;;
esac

case $2 in
  "db")
    db_args="--database test.db"
    ;;

  *)
    db_args="--database :flwr-in-memory-state:"
    ;;
esac

timeout 2m flower-server $server_args 0.0.0.0:8080 $db_args &
server_pid=$!
sleep 3

python client.py $client_args &
client1_pid=$!
sleep 3

python client.py $client_args &
client2_pid=$!
sleep 3

timeout 2m python driver.py &
pid=$!

wait $pid
res=$?

if [[ "$res" = "0" ]];
  then echo "Training worked correctly" && kill $client1_pid && kill $client2_pid && kill $server_pid;
  else echo "Training had an issue" && exit 1;
fi

