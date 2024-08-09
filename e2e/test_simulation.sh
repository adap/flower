#!/bin/bash
set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <framework>"
  exit 1
fi

framework=$1

case "$framework" in
  framework-pandas)
    server_app="server:app"
    client_app="client:app"
    app_dir="./"
    ;;
  *)
    server_app="server:app"
    client_app="${framework}.client:app"
    app_dir="./.."
    ;;
esac

echo flower-simulation --server-app $server_app --client-app $client_app --num-supernodes 2 --app-dir $app_dir

timeout 2m flower-simulation --server-app $server_app --client-app $client_app --num-supernodes 2 --app-dir $app_dir &
pid=$!

wait $pid
res=$?

if [[ "$res" = "0" ]];
  then echo "Training worked correctly";
  else echo "Training had an issue" && exit 1;
fi
