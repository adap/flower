#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

ROOT=$(pwd)

echo "Building baseline docs"
cd "$ROOT"
./dev/build-baseline-docs.sh

cd "$ROOT"
python dev/build-example-docs.py

cd "$ROOT"
./datasets/dev/build-flwr-datasets-docs.sh

cd "$ROOT"
cd intelligence/docs
make html

cd "$ROOT"
cd framework/docs 

if [ "$1" = true ]; then
    ./build-versioned-docs.sh
else
    make html
fi
