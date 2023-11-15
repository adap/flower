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

timeout 2m flower-server --grpc-bidi --grpc-bidi-fleet-api-address 0.0.0.0:8080 &
sleep 3

python client.py &
sleep 3

python client.py &
sleep 3

timeout 2m python $driver_file &
pid=$!

wait $pid
res=$?

if [[ "$res" = "0" ]];
  then echo "Training worked correctly" && pkill python;
  else echo "Training had an issue" && exit 1;
fi

