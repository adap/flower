#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../../

taplo fmt

# Python
python -m flwr_tool.check_copyright framework/src/py/flwr
python -m flwr_tool.init_py_fix framework/src/py/flwr
python -m isort --skip framework/src/py/flwr/proto framework/src/py
python -m black -q --exclude framework/src/py/flwr/proto framework/src/py
python -m docformatter -i -r framework/src/py/flwr -e framework/src/py/flwr/proto
python -m docformatter -i -r framework/src/py/flwr_tool
python -m ruff check --fix framework/src/py/flwr

# Protos
find framework/src/proto/flwr/proto -name *.proto | grep "\.proto" | xargs clang-format -i

# Examples
python -m black -q examples
python -m docformatter -i -r examples

# Benchmarks
python -m isort benchmarks
python -m black -q benchmarks
python -m docformatter -i -r benchmarks

# E2E
python -m isort e2e
python -m black -q e2e
python -m docformatter -i -r e2e

# Notebooks
python -m black --ipynb -q framework/doc/source/*.ipynb
KEYS="metadata.celltoolbar metadata.language_info metadata.toc metadata.notify_time metadata.varInspector metadata.accelerator metadata.vscode cell.metadata.id cell.metadata.heading_collapsed cell.metadata.hidden cell.metadata.code_folding cell.metadata.tags cell.metadata.init_cell cell.metadata.vscode cell.metadata.pycharm"
python -m nbstripout framework/doc/source/*.ipynb --extra-keys "$KEYS"
python -m nbstripout examples/*/*.ipynb --extra-keys "$KEYS"

# Markdown
python -m mdformat --number framework/doc/source examples

# RST
docstrfmt framework/doc/source
