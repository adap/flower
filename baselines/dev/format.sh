#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

python -m isort flwr_baselines
python -m black -q flwr_baselines
python -m docformatter -i -r flwr_baselines
