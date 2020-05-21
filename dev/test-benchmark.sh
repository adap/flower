#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

echo "=== test-benchmark.sh ==="

isort --check-only -rc src/flower_benchmark  && echo "- isort:  done" &&
black -q --check src/flower_benchmark           && echo "- black:  done" &&
# mypy is covered by test.sh
pylint src/flower_benchmark                  && echo "- pylint: done" &&
pytest -q src/flower_benchmark                  && echo "- pytest: done" &&
echo "- All Python checks passed"
