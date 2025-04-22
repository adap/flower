#!/bin/bash
# Build all docs
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

ROOT=$(pwd)

echo "Updating HTML theme options"
cd "$ROOT"
python dev/update-html-themes.py

echo "Building baseline docs"
cd "$ROOT"
./dev/build-baseline-docs.sh

cd "$ROOT"
python dev/build-example-docs.py

cd "$ROOT"
./datasets/dev/build-flwr-datasets-docs.sh

cd "$ROOT"
./framework/dev/build-docs.sh $1
