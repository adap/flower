#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Purpose of this script is to evaluate if the user changed the proto definitions
# but did not recompile or commit the new proto python files

# Recompile protos
python -m flower_tools.grpc > /dev/null 2>&1

# Fail if user forgot to recompile
CHANGED=$(git diff --name-only HEAD src/flower/proto)

if [ -n "$CHANGED" ]; then
    echo "Changes detected"
    exit 1
fi

echo "No changes detected"
exit 0
