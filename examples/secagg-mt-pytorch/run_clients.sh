#!/bin/bash

# Number of client processes to start
N=4 # Replace with your desired value

# Start N client processes
for i in $(seq 1 $N)
do
  nohup python client.py &
  sleep 1
done
