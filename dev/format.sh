#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Python
isort -s src/py/flwr/proto -rc src/py
black -q --exclude src/py/flwr/proto src/py
docformatter -i -r src/py/flwr -e src/py/flwr/proto
docformatter -i -r src/py/flwr_experimental

# Examples
isort -rc examples
black -q examples
docformatter -i -r examples
