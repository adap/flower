#!/bin/bash
# Test benchmarks and examples
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

echo "=== test.sh ==="

echo "- Start Python checks"

echo "- isort: start"
python -m isort --check-only ../benchmarks
echo "- isort: done"

echo "- black: start"
python -m black --check ../benchmarks ../examples
echo "- black: done"

echo "- All Python checks passed"

echo "- Start Markdown checks"

echo "- mdformat: start"
python -m mdformat --check --number ../examples
echo "- mdformat: done"

echo "- All Markdown checks passed"

echo "- Start TOML checks"

echo "- taplo: start"
taplo fmt --check ../benchmarks ../examples
echo "- taplo: done"

echo "- All TOML checks passed"
