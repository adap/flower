#!/bin/bash
set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <framework>"
  exit 1
fi

framework=$1

timeout 2m flower-simulation --server-app server:app --client-app ${framework}.client:app --num-supernodes 2 --app-dir ./.. &
pid=$!

wait $pid
res=$?

if [[ "$res" = "0" ]];
  then echo "Training worked correctly";
  else echo "Training had an issue" && exit 1;
fi
