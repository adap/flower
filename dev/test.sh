#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

echo "=== test.sh ==="

clang-format --Werror --dry-run src/proto/flwr/proto/*                    && echo "- clang-format:  done" &&
python -m isort --check-only --skip src/py/flwr/proto src/py/flwr         && echo "- isort:         done" &&
python -m black --exclude "src\/py\/flwr\/proto" --check src/py/flwr      && echo "- black:         done" &&
python -m flwr_tool.init_py_check src/py/flwr src/py/flwr_tool            && echo "- init_py_check: done" &&
python -m docformatter -c -r src/py/flwr -e src/py/flwr/proto             && echo "- docformatter:  done" &&
python -m mypy src/py                                                     && echo "- mypy:          done" &&
python -m pylint --ignore=src/py/flwr/proto src/py/flwr                   && echo "- pylint:        done" &&
python -m flake8 src/py/flwr                                              && echo "- flake8:        done" &&
python -m pytest --cov=src/py/flwr                                        && echo "- pytest:        done" &&
echo "- All Python checks passed"
