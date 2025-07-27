#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../


echo "Format code and run all test scripts"

./dev/format.sh
./baselines/dev/format.sh
./framework/dev/format.sh

./dev/test.sh
./baselines/dev/test.sh
./framework/dev/test.sh

./framework/dev/test-tool.sh
