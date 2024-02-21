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
    rest_arg=""
    server_address="127.0.0.1:9092"
    db_arg="--database :flwr-in-memory-state:"
    server_auth="--require-client-authentication ../client-auth/client_public_keys.csv ../client-auth/server_credential.pub ../client-auth/server_credential"
    client_auth_1="--authentication-keys ../client-auth/client_credential_1.pub ../client-auth/client_credential_1"
    client_auth_2="--authentication-keys ../client-auth/client_credential_2.pub ../client-auth/client_credential_2"
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

timeout 2m python driver.py &
pid=$!

wait $pid
res=$?

if [[ "$res" = "0" ]];
  then echo "Training worked correctly"; kill $cl1_pid; kill $cl2_pid; kill $sl_pid;
  else echo "Training had an issue" && exit 1;
fi

