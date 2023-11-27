#!/bin/bash
set -e

case "$1" in
  bare-https)
    ./generate.sh
    server_arg="--certificates certificates/ca.crt certificates/server.pem certificates/server.key"
    client_arg="--certificates certificates/ca.crt"
    ;;
  *)
    server_arg="--insecure"
    client_arg="--insecure"
    ;;
esac

timeout 2m flower-server $server_arg &
sleep 3

timeout 2m flower-client $client_arg --callable client:flower --server 127.0.0.1:9092 &
sleep 3

timeout 2m flower-client $client_arg --callable client:flower --server 127.0.0.1:9092 &
sleep 3

timeout 2m python driver.py &
pid=$!

wait $pid
res=$?

if [[ "$res" = "0" ]];
  then echo "Training worked correctly" && pkill flower-client && pkill flower-server;
  else echo "Training had an issue" && exit 1;
fi

