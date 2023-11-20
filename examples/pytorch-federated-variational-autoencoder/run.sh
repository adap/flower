#!/bin/bash

# Run server in the background
poetry run python3 server.py &

# Sleep for a short time to ensure the server has started before clients
sleep 2

# Run multiple clients in the background
poetry run python3 client.py &
poetry run python3 client.py
