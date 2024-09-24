#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

echo "=== test.sh ==="

echo "- Start Python checks"

echo "- clang-format:  start"
clang-format --Werror --dry-run src/proto/flwr/proto/*
echo "- clang-format:  done"

echo "- isort: start"
python -m isort --check-only --skip src/py/flwr/proto src/py/flwr benchmarks e2e
echo "- isort: done"

echo "- black: start"
python -m black --exclude "src\/py\/flwr\/proto" --check src/py/flwr benchmarks examples e2e
echo "- black: done"

echo "- init_py_check: start"
python -m flwr_tool.init_py_check src/py/flwr src/py/flwr_tool
echo "- init_py_check: done"

echo "- docformatter: start"
python -m docformatter -c -r src/py/flwr e2e -e src/py/flwr/proto
echo "- docformatter:  done"

echo "- ruff: start"
python -m ruff check src/py/flwr
echo "- ruff: done"

echo "- mypy: start"
python -m mypy src/py
echo "- mypy: done"

echo "- pylint: start"
python -m pylint --ignore=src/py/flwr/proto src/py/flwr
echo "- pylint: done"

echo "- flake8: start"
python -m flake8 src/py/flwr
echo "- flake8: done"

echo "- pytest: start"
python -m pytest --cov=src/py/flwr
echo "- pytest: done"

echo "- All Python checks passed"

echo "- Start Markdown checks"

echo "- mdformat: start"
python -m mdformat --check --number doc/source examples
echo "- mdformat: done"

echo "- All Markdown checks passed"

echo "- Start license checks"

echo "- copyright: start"
python -m flwr_tool.check_copyright src/py/flwr
echo "- copyright: done"

echo "- licensecheck: start"
python -m licensecheck -u poetry --fail-licenses gpl --zero
echo "- licensecheck: done"

echo "- All license checks passed"
