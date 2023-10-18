#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Append path to PYTHONPATH that makes flwr_tool.init_py_check discoverable
PARENT_DIR=$(dirname "$(pwd)") # Go one dir up from flower/datasets
export PYTHONPATH="${PYTHONPATH}:${PARENT_DIR}/src/py"

echo "=== test.sh ==="

echo "- Start Python checks"

echo "- isort: start"
python -m isort --check-only flwr_datasets/
echo "- isort: done"

echo "- black: start"
python -m black --check flwr_datasets/
echo "- black: done"

echo "- init_py_check: start"
python -m flwr_tool.init_py_check flwr_datasets/
echo "- init_py_check: done"

echo "- docformatter: start"
python -m docformatter -c -r flwr_datasets/
echo "- docformatter:  done"

echo "- ruff: start"
python -m ruff check flwr_datasets/
echo "- ruff: done"

echo "- mypy: start"
python -m mypy flwr_datasets/
echo "- mypy: done"

echo "- pylint: start"
python -m pylint flwr_datasets/
echo "- pylint: done"

echo "- flake8: start"
python -m flake8 flwr_datasets/
echo "- flake8: done"

echo "- pytest: start"
python -m pytest flwr_datasets/
echo "- pytest: done"

echo "- All Python checks passed"
