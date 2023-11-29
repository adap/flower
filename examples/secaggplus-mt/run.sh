#!/bin/bash
# Kill any currently running client.py processes
pkill -f 'flower-client'

# Kill any currently running flower-server processes
pkill -f 'flower-server'

# Start the flower server
echo "Starting flower server in background..."
flower-server --insecure > /dev/null 2>&1 &
sleep 2

# Number of client processes to start
N=5 # Replace with your desired value

echo "Starting $N clients in background..."

# Start N client processes
for i in $(seq 1 $N)
do
  flower-client --insecure --callable client:flower > /dev/null 2>&1 &
  sleep 0.1
done

echo "Starting driver..."
python driver.py

echo "Clearing background processes..."

# Kill any currently running client.py processes
pkill -f 'flower-client'

# Kill any currently running flower-server processes
pkill -f 'flower-server'
