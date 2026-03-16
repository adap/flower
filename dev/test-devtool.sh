#!/bin/bash
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"

echo "=== test-devtool.sh ==="

echo "- isort: start"
python -m isort --check-only devtool
echo "- isort: done"

echo "- black: start"
python -m black --check devtool
echo "- black: done"

echo "- All devtool Python checks passed"
