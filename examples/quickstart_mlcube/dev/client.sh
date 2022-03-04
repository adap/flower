#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

if [ $# -eq 0 ]
  then
    echo "Please provide the name of the client workspace e.g. 'python client.py one'"
    exit 1
fi

poetry run python client.py $1
