#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

echo "=== test.sh ==="


# Default value (true)
RUN_FULL_TEST=${1:true}
echo "RUN_FULL_TEST: $RUN_FULL_TEST"

echo "- Start Python checks"

echo "- clang-format:  start"
clang-format --Werror --dry-run proto/flwr/proto/*
echo "- clang-format:  done"

echo "- isort: start"
if $RUN_FULL_TEST; then
    python -m isort --check-only --skip py/flwr/proto py/flwr e2e
else
    python -m isort --check-only --skip py/flwr/proto py/flwr
fi
echo "- isort: done"

echo "- black: start"
if $RUN_FULL_TEST; then
    python -m black --exclude "py\/flwr\/proto" --check py/flwr e2e
else
    python -m black --exclude "py\/flwr\/proto" --check py/flwr
fi
echo "- black: done"

echo "- init_py_check: start"
python -m devtool.init_py_check py/flwr py/flwr_tool
echo "- init_py_check: done"

echo "- docformatter: start"
if $RUN_FULL_TEST; then
    python -m docformatter -c -r py/flwr e2e -e py/flwr/proto
else
    python -m docformatter -c -r py/flwr -e py/flwr/proto
fi
echo "- docformatter:  done"

echo "- docsig: start"
docsig py/flwr
echo "- docsig:  done"

echo "- ruff: start"
python -m ruff check py/flwr --no-respect-gitignore
echo "- ruff: done"

echo "- mypy: start"
python -m mypy py
echo "- mypy: done"

echo "- pylint: start"
python -m pylint --ignore=py/flwr/proto py/flwr
echo "- pylint: done"

echo "- pytest: start"
python -m pytest --cov=py/flwr
echo "- pytest: done"

echo "- All Python checks passed"

echo "- Start Markdown checks"

if $RUN_FULL_TEST; then
    echo "- mdformat: start"
    python -m mdformat --check --number docs/source
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
    docstrfmt --check docs/source
    echo "- docstrfmt: done"
fi

echo "- All rST checks passed"

echo "- Start license checks"

if $RUN_FULL_TEST; then
    echo "- copyright: start"
    python -m devtool.check_copyright py/flwr
    echo "- copyright: done"
fi

echo "- licensecheck: start"
python -m licensecheck -u poetry --fail-licenses gpl --zero
echo "- licensecheck: done"

echo "- All license checks passed"
