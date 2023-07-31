#!/bin/bash
set -e

# run the first command in background and save output to a temporary file:
timeout 2m python server.py &
pid=$!
sleep 3

python client.py &
sleep 3

python client.py &
sleep 3

wait $pid
res=$?

if [[ "$res" = "0" ]];
  then echo "Training worked correctly";
  else echo "Training had an issue" && exit 1;
fi

