#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Python
isort -s src/py/flwr/proto -rc src
black -q --exclude src/py/flwr/proto src
docformatter -i -r src/py/flwr -e src/py/flwr/proto
