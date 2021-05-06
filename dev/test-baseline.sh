#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

echo "=== test-baseline.sh ==="

python -m isort --check-only src/py/flwr_experimental/baseline          && echo "- isort:  done" &&
python -m black --check src/py/flwr_experimental/baseline               && echo "- black:  done" &&
# mypy is covered by test.sh
# python -m pylint src/py/flwr_experimental/baseline                      && echo "- pylint: done" &&
# python -m pytest --durations=0 -v src/py/flwr_experimental/baseline     && echo "- pytest: done" &&
echo "- All Python checks passed"
