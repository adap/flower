#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

echo "=== test-example-tensorflow.sh ==="

isort --check-only -rc src/flwr_example/tensorflow  && echo "- isort:  done" &&
black -q --check src/flwr_example/tensorflow        && echo "- black:  done" &&
# mypy is covered by test.sh
pylint src/flwr_example/tensorflow                  && echo "- pylint: done" &&
pytest -q src/flwr_example/tensorflow               && echo "- pytest: done" &&
echo "- All Python checks passed"
