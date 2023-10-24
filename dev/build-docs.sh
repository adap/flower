#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

ROOT=`pwd`

cd $ROOT
./dev/build-baseline-docs.sh

cd $ROOT
./dev/update-examples.sh
cd examples/doc
make docs

cd $ROOT
cd datasets/doc
make docs

cd $ROOT
cd doc
./build-versioned-docs.sh
