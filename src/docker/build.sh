#!/bin/bash

# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
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
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../../

HASH=$(printf "$(git rev-parse HEAD)\n$(git diff | sha1sum)" | sha1sum | cut -c1-7)

rm -rf dist
python -m poetry build
docker build -f src/docker/default.Dockerfile -t flower:latest -t flower:$HASH .
docker build -f src/docker/sshd.Dockerfile --build-arg SSH_PUBLIC_KEY="$(cat docker/ssh_key.pub)" -t flower-sshd:latest -t flower-sshd:$HASH .
