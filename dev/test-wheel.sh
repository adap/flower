#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

echo "=== test-wheel.sh ==="

# Build the wheel first
echo "Building the wheel: start"
./dev/build.sh
echo "Building the wheel: done"

# Test
echo "Twine wheel check: start"
python -m twine check --strict ./dist/*
echo "Twine wheel check: done"

echo "- All wheel checks passed"
