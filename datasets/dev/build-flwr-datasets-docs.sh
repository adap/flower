#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

echo "Build Flower Datasets docs: Start"
cd doc
make docs
echo "Build Flower Datasets docs: Done"
