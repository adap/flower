#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

ROOT=$(pwd)

cd "$ROOT"
cd docs 

if [ "$1" = true ]; then
    ./build-versioned-docs.sh
else
    make html
fi
