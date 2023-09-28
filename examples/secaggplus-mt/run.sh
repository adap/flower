#!/bin/bash
# Kill any currently running client.py processes
pkill -f 'python client.py'

# Kill any currently running flower-server processes with --grpc-rere option
pkill -f 'flower-server --grpc-rere'

# Start the flower server
echo "Starting flower server in background..."
flower-server --grpc-rere > /dev/null 2>&1 &
sleep 2

# Number of client processes to start
N=5 # Replace with your desired value

echo "Starting $N clients in background..."

# Start N client processes
for i in $(seq 1 $N)
do
  python client.py > /dev/null 2>&1 &
  # python client.py &
  sleep 0.1
done

echo "Starting driver..."
python driver.py

echo "Clearing background processes..."

# Kill any currently running client.py processes
pkill -f 'python client.py'

# Kill any currently running flower-server processes with --grpc-rere option
pkill -f 'flower-server --grpc-rere'
