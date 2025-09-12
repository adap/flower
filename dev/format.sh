#!/bin/bash
# Format examples and benchmarks
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

taplo fmt

# Examples
# Find all source folders in examples/
src_args=$(find ../examples -mindepth 1 -maxdepth 2 -type d -printf '--src %p ')
python -m isort ../examples $src_args --settings-path .
python -m black -q ../examples
python -m docformatter -i -r ../examples

# Benchmarks
python -m isort ../benchmarks
python -m black -q ../benchmarks
python -m docformatter -i -r ../benchmarks

# Notebooks
KEYS="metadata.celltoolbar metadata.language_info metadata.toc metadata.notify_time metadata.varInspector metadata.accelerator metadata.vscode cell.metadata.id cell.metadata.heading_collapsed cell.metadata.hidden cell.metadata.code_folding cell.metadata.tags cell.metadata.init_cell cell.metadata.vscode cell.metadata.pycharm"
python -m nbstripout ../examples/*/*.ipynb --extra-keys "$KEYS"

# Markdown
python -m mdformat --number ../examples
