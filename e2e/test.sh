#!/bin/bash
set -e

case "$1" in
  pandas)
    server_file="server.py"
    ;;
  bare-https)
    ./generate.sh
    server_file="server.py"
    ;;
  *)
    server_file="../server.py"
    ;;
esac

# run the first command in background and save output to a temporary file:
timeout 2m python $server_file &
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

