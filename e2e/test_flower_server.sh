#!/bin/bash
set -e

case "$1" in
  pandas)
    driver_file="driver.py"
    ;;
  *)
    driver_file="../driver.py"
    ;;
esac

case "$2" in
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

case "$3" in
  "db")
    db_args="--database $(date +%s).db"
    ;;

  *)
    db_args="--database :flwr-in-memory-state:"
    ;;
esac

timeout 5m flower-server $server_args 0.0.0.0:8080 $db_args &
server_pid=$!
sleep 3

python client.py $client_args &
client1_pid=$!
sleep 3

python client.py $client_args &
client2_pid=$!
sleep 3

timeout 5m python $driver_file &
pid=$!

wait $pid
res=$?

if [[ "$res" = "0" ]];
  then 
    echo "Training worked correctly"
    sleep 3
    kill $client1_pid
    sleep 3
    kill $client2_pid
    sleep 3
    kill $server_pid
    exit 0
  else echo "Training had an issue" && exit 1;
fi

