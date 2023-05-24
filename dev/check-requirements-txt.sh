#!/bin/bash

# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Regenerate requirements.txt files for examples in case they changed
echo "Regenerate requirements.txt files in case they changed"
./dev/generate-requirements-txt.sh 2> /dev/null

# Fail if user forgot to sync requirements.txt and pyproject.toml
CHANGED=$(git diff --name-only HEAD examples)

if [ -n "$CHANGED" ]; then
    echo "Changes detected, requirements.txt and pyproject.toml is not synced. Please run the script dev/generate-requirements-txt."
    exit 1
fi

echo "No changes detected"
exit 0
