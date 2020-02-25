#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

isort --skip src/flower/proto --skip src/flower_examples --check-only -rc src  && echo "- isort:  done" &&
black --exclude "src\/(flower\/proto|flower_examples)" --check src             && echo "- black:  done" &&
mypy src                                                                       && echo "- mypy:   done" &&
pylint --ignore=src/flower/proto src/flower src/flower_tools                   && echo "- pylint: done" &&
pytest src/flower src/flower_tools                                             && echo "- pytest: done" &&
echo "- All Python checks passed"
