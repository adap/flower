#!/bin/bash
set -e

if [ "$1" = "e2e-bare-https" ]; then
  ./../generate.sh
fi

# run the first command in background and save output to a temporary file:
timeout 3m python server_app.py &
pid=$!
sleep 3

python client_app.py &
sleep 3

python client_app.py &
sleep 3

wait $pid
res=$?

if [[ "$res" = "0" ]];
  then echo "Training worked correctly";
  else echo "Training had an issue" && exit 1;
fi

