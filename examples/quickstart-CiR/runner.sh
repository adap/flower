#!/bin/bash

# Start the server in the first terminal
gnome-terminal -- bash -c "python server.py; read -p 'Press Enter to exit';"

# Wait for 2 seconds
sleep 2

num_clients=4

# Start the clients in a for loop
for i in $(seq 0 $((num_clients - 1))); do
    gnome-terminal -- bash -c "sleep $i; python client.py --client_idx $i; read -p 'Press Enter to exit';"
done
