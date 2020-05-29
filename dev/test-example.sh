#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

echo "=== test-example.sh ==="

isort --check-only -rc src/flower_example  && echo "- isort:  done" &&
black -q --check src/flower_example           && echo "- black:  done" &&
# mypy is covered by test.sh
pylint src/flower_example                  && echo "- pylint: done" &&
pytest -q src/flower_example                  && echo "- pytest: done" &&
echo "- All Python checks passed"
