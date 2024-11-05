#!/bin/bash

# Copyright 2022 Flower Labs GmbH. All Rights Reserved.
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

# Initialize variables
enterprise_extensions_path=""

# Function to display usage
function usage() {
    echo "Usage: $0 [--enterprise-extensions <path-to-enterprise-extensions>]"
    exit 1
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --enterprise-extensions)
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                enterprise_extensions_path="$2"
                shift 2
            else
                echo "Error: --enterprise-extensions requires a path as an argument."
                usage
            fi
            ;;
        *)
            echo "Unknown parameter: $1"
            usage
            ;;
    esac
done

# Display the path if provided
if [[ -n "$enterprise_extensions_path" ]]; then
    # Preparation with enterprise extensions
    echo "Building with enterprise extensions located at: $enterprise_extensions_path"
    (rm -r src/py/flwr_ee || true) &> /dev/null
    cp -r $enterprise_extensions_path src/py/
else
    # Preparation without enterprise extensions
    echo "Building without enterprise extensions."
fi

# Build commands
python -m poetry build

echo "Build complete."

# Clean up enterprise extensions
(rm -r src/py/flwr_ee || true) &> /dev/null
