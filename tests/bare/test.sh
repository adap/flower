#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# run the first command in background and save output to a temporary file:
timeout 20 python server.py &
pid=$!
sleep 3

python client.py 1 > /dev/null 2>&1 &

python client.py 2 > /dev/null 2>&1 &

wait $pid
res=$?

if [[ "$res" = "0" ]];
  then echo "Training worked correctly";
  else echo "Training had an issue" && exit 1;
fi

