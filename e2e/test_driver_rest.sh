#!/bin/bash
set -e

timeout 2m flower-server --rest --rest-fleet-api-address 0.0.0.0:8080 &
server_pid=$!
sleep 3

python client.py rest &
sleep 3

python client.py rest &
sleep 3

timeout 2m python driver.py &
pid=$!

wait $pid
res=$?

if [[ "$res" = "0" ]];
  then echo "Training worked correctly" && kill server_pid;
  else echo "Training had an issue" && exit 1;
fi

