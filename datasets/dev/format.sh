#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Python
echo "Formatting started: Python"
python -m isort flwr_datasets/
python -m black -q flwr_datasets/
python -m docformatter -i -r flwr_datasets/
python -m ruff check --fix flwr_datasets/
echo "Formatting done: Python"

# Notebooks
echo "Formatting started: Notebooks"
python -m black --ipynb -q doc/source/*.ipynb
KEYS="metadata.celltoolbar metadata.language_info metadata.toc metadata.notify_time metadata.varInspector metadata.accelerator metadata.vscode cell.metadata.id cell.metadata.heading_collapsed cell.metadata.hidden cell.metadata.code_folding cell.metadata.tags cell.metadata.init_cell cell.metadata.vscode cell.metadata.pycharm"
python -m nbstripout --keep-output doc/source/*.ipynb --extra-keys "$KEYS"
echo "Formatting done: Notebooks"
