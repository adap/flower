#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

isort --check-only -rc src/flower_benchmark  && echo "- isort:  done" &&
black --check src/flower_benchmark           && echo "- black:  done" &&
# mypy is covered by test.sh
pylint src/flower_benchmark                  && echo "- pylint: done" &&
pytest src/flower_benchmark                  && echo "- pytest: done" &&
echo "- All Python checks passed"
