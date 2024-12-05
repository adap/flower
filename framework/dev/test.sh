#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../../

echo "=== test.sh ==="

echo "- Start Python checks"

echo "- clang-format:  start"
clang-format --Werror --dry-run framework/src/proto/flwr/proto/*
echo "- clang-format:  done"

echo "- isort: start"
python -m isort --check-only --skip framework/src/py/flwr/proto framework/src/py/flwr benchmarks e2e
echo "- isort: done"

echo "- black: start"
python -m black --exclude "framework\/src\/py\/flwr\/proto" --check framework/src/py/flwr benchmarks examples e2e
echo "- black: done"

echo "- init_py_check: start"
python -m flwr_tool.init_py_check framework/src/py/flwr framework/src/py/flwr_tool
echo "- init_py_check: done"

echo "- docformatter: start"
python -m docformatter -c -r framework/src/py/flwr e2e -e framework/src/py/flwr/proto
echo "- docformatter:  done"

echo "- docsig: start"
docsig framework/src/py/flwr
echo "- docsig:  done"

echo "- ruff: start"
python -m ruff check framework/src/py/flwr
echo "- ruff: done"

echo "- mypy: start"
python -m mypy framework/src/py
echo "- mypy: done"

echo "- pylint: start"
python -m pylint --ignore=framework/src/py/flwr/proto framework/src/py/flwr
echo "- pylint: done"

echo "- flake8: start"
python -m flake8 framework/src/py/flwr
echo "- flake8: done"

echo "- pytest: start"
python -m pytest --cov=framework/src/py/flwr
echo "- pytest: done"

echo "- All Python checks passed"

echo "- Start Markdown checks"

echo "- mdformat: start"
python -m mdformat --check --number framework/doc/source examples
echo "- mdformat: done"

echo "- All Markdown checks passed"

echo "- Start TOML checks"

echo "- taplo: start"
taplo fmt --check
echo "- taplo: done"

echo "- All TOML checks passed"

echo "- Start rST checks"

echo "- docstrfmt: start"
docstrfmt --check framework/doc/source
echo "- docstrfmt: done"

echo "- All rST checks passed"

echo "- Start license checks"

echo "- copyright: start"
python -m flwr_tool.check_copyright framework/src/py/flwr
echo "- copyright: done"

echo "- licensecheck: start"
python -m licensecheck -u poetry --fail-licenses gpl --zero
echo "- licensecheck: done"

echo "- All license checks passed"
