#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Python
echo "Formatting started"
python -m isort flwr_datasets/
python -m black -q flwr_datasets/
python -m docformatter -i -r flwr_datasets/
python -m ruff check --fix flwr_datasets/
echo "Formatting done"
