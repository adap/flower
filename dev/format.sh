#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Python
python -m isort --skip src/py/flwr/proto src/py
python -m black -q --exclude src/py/flwr/proto src/py
python -m docformatter -i -r src/py/flwr -e src/py/flwr/proto
python -m docformatter -i -r src/py/flwr_experimental

# Protos
find src/proto/flwr/proto -name *.proto | grep "\.proto" | xargs clang-format-10 -i

# Examples
python -m black -q examples
python -m docformatter -i -r examples

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../examples

cd embedded_devices && python -m isort . && cd ..
cd pytorch_from_centralized_to_federated && python -m isort . && cd ..
cd quickstart_pytorch && python -m isort . && cd ..
cd quickstart_tensorflow && python -m isort . && cd ..
cd simulation && python -m isort . && cd ..
