#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

echo "=== test.sh ==="


# Default value (true)
RUN_FULL_TEST=${1:true}
echo "RUN_FULL_TEST: $RUN_FULL_TEST"

echo "- Start Python checks"

echo "- clang-format:  start"
clang-format --Werror --dry-run src/proto/flwr/proto/*
echo "- clang-format:  done"

echo "- isort: start"
if $RUN_FULL_TEST; then
    python -m isort --check-only --skip src/py/flwr/proto src/py/flwr benchmarks e2e
else
    python -m isort --check-only --skip src/py/flwr/proto src/py/flwr
fi
echo "- isort: done"

echo "- black: start"
if $RUN_FULL_TEST; then
    python -m black --exclude "src\/py\/flwr\/proto" --check src/py/flwr benchmarks examples e2e
else
    python -m black --exclude "src\/py\/flwr\/proto" --check src/py/flwr
fi
echo "- black: done"

echo "- init_py_check: start"
python -m flwr_tool.init_py_check src/py/flwr src/py/flwr_tool
echo "- init_py_check: done"

echo "- docformatter: start"
python -m docformatter -c -r src/py/flwr e2e -e src/py/flwr/proto
echo "- docformatter:  done"

echo "- docsig: start"
docsig src/py/flwr
echo "- docsig:  done"

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

if $RUN_FULL_TEST; then
    echo "- mdformat: start"
    python -m mdformat --check --number framework/docs/source examples
    echo "- mdformat: done"
fi

echo "- All Markdown checks passed"

echo "- Start TOML checks"

echo "- taplo: start"
taplo fmt --check
echo "- taplo: done"

echo "- All TOML checks passed"

echo "- Start rST checks"

if $RUN_FULL_TEST; then
    echo "- docstrfmt: start"
    docstrfmt --check framework/docs/source
    echo "- docstrfmt: done"
fi

echo "- All rST checks passed"

echo "- Start license checks"

if $RUN_FULL_TEST; then
    echo "- copyright: start"
    python -m flwr_tool.check_copyright src/py/flwr
    echo "- copyright: done"
fi

echo "- licensecheck: start"
python -m licensecheck -u poetry --fail-licenses gpl --zero
echo "- licensecheck: done"

echo "- All license checks passed"
