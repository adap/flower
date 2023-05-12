#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# run the first command in background and save output to a temporary file:
timeout 2m python server.py &
sleep 3

python client.py 1 > /dev/null 2>&1 &
sleep 3

if [[ $(ps aux | grep "[p]ython client.py 1" | awk '{ print $2 }') ]];
  then echo "Client process 1 started correctly";
  else echo "Client process 1 crashed" && exit 1;
fi

python client.py 2 > /dev/null 2>&1 &
sleep 3

if [[ $(ps aux | grep "[p]ython client.py 2" | awk '{ print $2 }') ]];
  then echo "Client process 2 started correctly";
  else echo "Client process 2 crashed" && exit 1;
fi

if [[ $(ps aux | grep "[p]ython server.py" | awk '{ print $2 }') ]];
  then echo "Server process started correctly";
  else echo "Server process crashed" && exit 1;
fi

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# Wait for all background processes to complete
wait $pid
res=$?

if [[ "$res" = "0" ]];
  then echo "Training worked correctly";
  else echo "Training had an issue" && exit 1;
fi
