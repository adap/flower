#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

echo "=== test-tool.sh ==="

isort --check-only -rc src/flower_tool  && echo "- isort:  done" &&
black -q --check src/flower_tool           && echo "- black:  done" &&
# mypy is covered by test.sh
pylint src/flower_tool                  && echo "- pylint: done" &&
pytest -q src/flower_tool                  && echo "- pytest: done" &&
echo "- All Python checks passed"
