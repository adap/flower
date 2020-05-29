#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

echo "=== test.sh ==="

isort --skip src/flower/proto --check-only -rc src/flower  && echo "- isort:  done" &&
black -q --exclude "src\/flower\/proto" --check src/flower    && echo "- black:  done" &&
mypy src                                                   && echo "- mypy:   done" &&
pylint --ignore=src/flower/proto src/flower                && echo "- pylint: done" &&
pytest -q src/flower                                          && echo "- pytest: done" &&
echo "- All Python checks passed"
