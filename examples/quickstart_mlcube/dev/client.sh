#!/bin/bash

set -e

echo "Prepare client $1"

if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
    exit 1
fi

# Create client directory
mkdir -p clients/client_$1

# Copy mlcube and client
cp -r mlcube clients/client_$1/
cp -r client.py clients/client_$1/
cp -r mlcube_utils.py clients/client_$1/

echo "Start client $1"
# Start client Python process in background
poetry run python clients/client_$1/client.py
