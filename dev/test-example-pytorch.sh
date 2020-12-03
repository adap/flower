#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

echo "=== test-example-pytorch.sh ==="

isort --check-only -rc src/py/flwr_example/pytorch_cifar  && echo "- isort:  done" &&
black -q --check src/py/flwr_example/pytorch_cifar        && echo "- black:  done" &&
# mypy is covered by test.sh
pylint src/py/flwr_example/pytorch_cifar                  && echo "- pylint: done" &&
pytest -q src/py/flwr_example/pytorch_cifar               && echo "- pytest: done" &&
echo "- All Python checks passed"
