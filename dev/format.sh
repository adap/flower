#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Python
isort -s src/flwr/proto -rc src
black -q --exclude src/flwr/proto src
docformatter -i -r src/flwr -e src/flwr/proto