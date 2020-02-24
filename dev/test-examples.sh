#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

isort -s src/flower/proto --check-only -rc src/flower_examples  && echo "- isort:  done" &&
black --exclude src/flower/proto --check src/flower_examples    && echo "- black:  done" &&
mypy src/flower_examples                                        && echo "- mypy:   done" &&
pylint --ignore=src/flower/proto src/flower_examples            && echo "- pylint: done" &&
pytest src/flower_examples                                      && echo "- pytest: done" &&
echo "- All Python checks passed"
