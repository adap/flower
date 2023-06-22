#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

echo "=== test.sh ==="

echo "- clang-format:  start" &&
clang-format --Werror --dry-run src/proto/flwr/proto/* &&
echo "- clang-format:  done" &&

echo "- isort: start" &&
python -m isort --check-only --skip src/py/flwr/proto src/py/flwr &&
nbqa isort --check-only doc/source/tutorial examples &&
echo "- isort: done" &&

echo "- black: start" &&
python -m black --exclude "src\/py\/flwr\/proto" --check src/py/flwr &&
nbqa black --check doc/source/tutorial examples &&
echo "- black: done" &&

echo "- init_py_check: start" &&
python -m flwr_tool.init_py_check src/py/flwr src/py/flwr_tool &&
echo "- init_py_check: done" &&

echo "- docformatter: start" &&
python -m docformatter -c -r src/py/flwr -e src/py/flwr/proto &&
nbqa docformatter -c -r doc/source/tutorial examples &&
echo "- docformatter:  done" &&

echo "- ruff: start" &&
python -m ruff check src/py/flwr &&
nbqa ruff doc/source/tutorial examples --extend-ignore=D100,D101,D102,D103,D104,D105,D106,D107 &&
echo "- ruff: done" &&

echo "- mypy: start" &&
python -m mypy src/py &&
nbqa mypy doc/source/tutorial examples &&
echo "- mypy: done" &&

echo "- pylint: start" &&
python -m pylint --ignore=src/py/flwr/proto src/py/flwr &&
nbqa pylint doc/source/tutorial examples &&
echo "- pylint: done" &&

echo "- flake8: start" &&
python -m flake8 src/py/flwr &&
nbqa flake8 doc/source/tutorial examples &&
echo "- flake8: done" &&

echo "- pytest: start" &&
python -m pytest --cov=src/py/flwr &&
echo "- pytest: done" &&

echo "- All Python checks passed"
