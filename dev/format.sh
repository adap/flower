#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Python
python -m isort -s src/py/flwr/proto -rc src/py
python -m black -q --exclude src/py/flwr/proto src/py
python -m docformatter -i -r src/py/flwr -e src/py/flwr/proto
python -m docformatter -i -r src/py/flwr_experimental

# Protos
find src/proto/flwr/proto -name *.proto | grep "\.proto" | xargs clang-format-10 -i

# Examples
python -m isort -rc examples
python -m black -q examples
python -m docformatter -i -r examples
