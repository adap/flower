#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../


echo "Format code and run all test scripts"

./dev/format.sh
./baselines/dev/format.sh

./dev/test.sh
./baselines/dev/test.sh

./dev/test-tool.sh
