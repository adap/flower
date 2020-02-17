#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

isort --check-only -rc src && echo "- isort:  done" &&
black --check src          && echo "- black:  done" &&
mypy src                   && echo "- mypy:   done" &&
pylint src/flower          && echo "- pylint: done" &&
pytest src/flower          && echo "- pytest: done" &&
echo "- All Python checks passed"
