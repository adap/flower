#!/bin/bash
set +e # continue if one script fails
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/


echo "Format code and run all test scripts"
./format.sh
./test.sh
./test-benchmark.sh
./test-logserver.sh
./test-example.sh
./test-ops.sh
./test-tool.sh
