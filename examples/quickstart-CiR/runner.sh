#!/bin/bash

# Start the server in the first terminal
gnome-terminal -- bash -c "python server.py; read -p 'Press Enter to exit';"

# Wait for 2 seconds
sleep 2

# Start the first client in the second terminal
gnome-terminal -- bash -c "python client.py; read -p 'Press Enter to exit';"

# Start the second client in the third terminal
gnome-terminal -- bash -c "python client.py; read -p 'Press Enter to exit';"
