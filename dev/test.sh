#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

echo "=== test.sh ==="

isort --skip src/py/flwr/proto --check-only -rc src/py/flwr    && echo "- isort:  done" &&
black -q --exclude "src\/py\/flwr\/proto" --check src/py/flwr  && echo "- black:  done" &&
mypy src/py                                                    && echo "- mypy:   done" &&
pylint --ignore=src/py/flwr/proto src/py/flwr                  && echo "- pylint: done" &&
pytest -q src/py/flwr                                          && echo "- pytest: done" &&
echo "- All Python checks passed"
