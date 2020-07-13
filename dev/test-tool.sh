#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

echo "=== test-tool.sh ==="

isort --check-only -rc src/flwr_tool  && echo "- isort:  done" &&
black -q --check src/flwr_tool           && echo "- black:  done" &&
# mypy is covered by test.sh
pylint src/flwr_tool                  && echo "- pylint: done" &&
pytest -q src/flwr_tool                  && echo "- pytest: done" &&
echo "- All Python checks passed"
