#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Python
isort -s src/flower/proto -rc src
black -q --exclude src/flower/proto src
