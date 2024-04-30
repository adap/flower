#!/bin/bash
set -e

case "$1" in
  pandas)
    server_dir="./"
    ;;
  bare-https)
    ./generate.sh
    server_arg="--certificates certificates/ca.crt certificates/server.pem certificates/server.key"
    client_arg="--root-certificates certificates/ca.crt"
    server_dir="./"
    ;;
  *)
    server_arg="--insecure"
    client_arg="--insecure"
    server_dir="./.."
    ;;
esac

case "$2" in
  rest)
    rest_arg="--rest"
    server_address="http://localhost:9093"
    db_arg="--database :flwr-in-memory-state:"
    server_auth=""
    client_auth_1=""
    client_auth_2=""
    ;;
  sqlite)
    rest_arg=""
    server_address="127.0.0.1:9092"
    db_arg="--database $(date +%s).db"
    server_auth=""
    client_auth_1=""
    client_auth_2=""
    ;;
  client-auth)
    ./generate.sh
    rest_arg=""
    server_address="127.0.0.1:9092"
    db_arg="--database :flwr-in-memory-state:"
    server_arg="--certificates certificates/ca.crt certificates/server.pem certificates/server.key"
    client_arg="--root-certificates certificates/ca.crt"
    server_auth="--require-client-authentication keys/client_public_keys.csv keys/server_credentials keys/server_credentials.pub"
    client_auth_1="--authentication-keys keys/client_credentials_1 keys/client_credentials_1.pub"
    client_auth_2="--authentication-keys keys/client_credentials_2 keys/client_credentials_2.pub"
    ;;
  *)
    rest_arg=""
    server_address="127.0.0.1:9092"
    db_arg="--database :flwr-in-memory-state:"
    server_auth=""
    client_auth_1=""
    client_auth_2=""
    ;;
esac

timeout 2m flower-superlink $server_arg $db_arg $rest_arg $server_auth &
sl_pid=$!
sleep 3

timeout 2m flower-client-app client:app $client_arg $rest_arg --server $server_address $client_auth_1 &
cl1_pid=$!
sleep 3

timeout 2m flower-client-app client:app $client_arg $rest_arg --server $server_address $client_auth_2 &
cl2_pid=$!
sleep 3

timeout 2m flower-server-app server:app $client_arg --dir $server_dir --server $server_address &
pid=$!

wait $pid
res=$?

if [[ "$res" = "0" ]];
  then echo "Training worked correctly"; kill $cl1_pid; kill $cl2_pid; kill $sl_pid;
  else echo "Training had an issue" && exit 1;
fi

