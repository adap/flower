#!/bin/bash

set -e

if [ $# -eq 0 ]
  then
    echo "Please provide the name of the client workspace e.g. 'python client.py one'"
    exit 1
fi

poetry run python client.py $1
