#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Python
python -m isort --skip src/py/flwr/proto src/py
python -m black -q --exclude src/py/flwr/proto src/py
python -m docformatter -i -r src/py/flwr -e src/py/flwr/proto
python -m docformatter -i -r src/py/flwr_experimental
python -m black -q baselines/flwr_baselines/experiments/fedbn_convergence_rate
python -m docformatter -i -r baselines/flwr_baselines/experiments/fedbn_convergence_rate
python -m isort baselines/flwr_baselines/experiments/fedbn_convergence_rate

# Protos
find src/proto/flwr/proto -name *.proto | grep "\.proto" | xargs clang-format-10 -i

# Examples
python -m black -q examples
python -m docformatter -i -r examples
