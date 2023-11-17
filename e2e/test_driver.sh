#!/bin/bash
set -e

case "$1" in
  bare-https)
    ./generate.sh
    cert_arg="--certificates certificates/ca.crt certificates/server.pem certificates/server.key"
    ;;
  *)
    cert_arg="--insecure"
    ;;
esac

timeout 2m flower-server $cert_arg --grpc-bidi --grpc-bidi-fleet-api-address 0.0.0.0:8080 &
sleep 3

python client.py &
sleep 3

python client.py &
sleep 3

timeout 2m python driver.py &
pid=$!

wait $pid
res=$?

if [[ "$res" = "0" ]];
  then echo "Training worked correctly" && pkill python;
  else echo "Training had an issue" && exit 1;
fi

