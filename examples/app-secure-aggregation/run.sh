#!/bin/bash
# Kill any currently running client.py processes
pkill -f 'flower-client-app'

# Kill any currently running flower-superlink processes
pkill -f 'flower-superlink'

# Start the flower server
echo "Starting flower server in background..."
flower-superlink --insecure > /dev/null 2>&1 &
sleep 2

# Number of client processes to start
N=5 # Replace with your desired value

echo "Starting $N ClientApps in background..."

# Start N client processes
for i in $(seq 1 $N)
do
  flower-client-app --insecure client:app > /dev/null 2>&1 &
  sleep 0.1
done

echo "Starting ServerApp..."
flower-server-app --insecure server:app --verbose

echo "Clearing background processes..."

# Kill any currently running client.py processes
pkill -f 'flower-client-app'

# Kill any currently running flower-superlink processes
pkill -f 'flower-superlink'
