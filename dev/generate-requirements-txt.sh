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

# Purpose of this script is to regenerate requirements.txt
for path in $(find ./examples -type f -name 'pyproject.toml' | sed -E 's|/[^/]+$||' |sort -u)
do
    if [ -f "$path/requirements.txt" ]; then
        cd $path &&

        sed -n '/\[tool.poetry.dependencies/q;p' pyproject.toml > pyproject.new.toml &&
        echo '[tool.poetry.dependencies]\npython = ">=3.8,<3.11"' >> pyproject.new.toml &&
        mv pyproject.new.toml pyproject.toml &&
        rm -rf poetry.lock &&

        echo -e "\nRunning poeareq for example in ${path}" &&
        poeareq "requirements.txt" &&
    
        cd ../../
    else 
        echo "$path/requirements.txt does not exist."
    fi 
done
