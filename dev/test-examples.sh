#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

isort --check-only -rc src/flower_examples  && echo "- isort:  done" &&
black --check src/flower_examples           && echo "- black:  done" &&
# mypy is covered by test.sh
pylint src/flower_examples                  && echo "- pylint: done" &&
pytest src/flower_examples                  && echo "- pytest: done" &&
echo "- All Python checks passed"
