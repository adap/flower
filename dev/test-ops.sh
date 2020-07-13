#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

echo "=== test-ops.sh ==="

isort --check-only -rc src/flwr_experimental/ops  && echo "- isort:  done" &&
black -q --check src/flwr_experimental/ops           && echo "- black:  done" &&
# mypy is covered by test.sh
pylint src/flwr_experimental/ops                  && echo "- pylint: done" &&
pytest -q src/flwr_experimental/ops                  && echo "- pytest: done" &&
echo "- All Python checks passed"
