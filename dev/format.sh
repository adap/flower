#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Python
python -m isort --skip src/py/flwr/proto src/py
python -m black -q --exclude src/py/flwr/proto src/py
python -m docformatter -i -r src/py/flwr -e src/py/flwr/proto
python -m docformatter -i -r src/py/flwr_tool

# Protos
find src/proto/flwr/proto -name *.proto | grep "\.proto" | xargs clang-format -i

# Examples
python -m black -q examples
python -m docformatter -i -r examples


# Notebooks
KEYS="metadata.celltoolbar metadata.language_info metadata.toc metadata.notify_time metadata.varInspector metadata.accelerator metadata.vscode cell.metadata.id cell.metadata.heading_collapsed cell.metadata.hidden cell.metadata.code_folding cell.metadata.tags cell.metadata.init_cell cell.metadata.vscode"
python -m nbstripout doc/source/tutorial/*.ipynb --extra-keys "$KEYS"
python -m nbstripout examples/*/*.ipynb --extra-keys "$KEYS"
