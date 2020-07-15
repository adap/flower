#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/


echo "Format code and run all test scripts"
./format.sh
./test.sh
./test-baseline.sh
./test-example-pytorch.sh
./test-example-tensorflow.sh
./test-logserver.sh
./test-ops.sh
./test-tool.sh
