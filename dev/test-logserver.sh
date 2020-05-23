#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

echo "=== test-logserver.sh ==="

isort --check-only -rc src/flower_logserver  && echo "- isort:  done" &&
black -q --check src/flower_logserver           && echo "- black:  done" &&
# mypy is covered by test.sh
pylint src/flower_logserver                  && echo "- pylint: done" &&
pytest -q src/flower_logserver                  && echo "- pytest: done" &&
echo "- All Python checks passed"
