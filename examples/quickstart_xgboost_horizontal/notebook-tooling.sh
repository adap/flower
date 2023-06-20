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

#set -e
#cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

nbqa black code_horizontal.ipynb
nbqa isort code_horizontal.ipynb
nbqa flake8 code_horizontal.ipynb --extend-ignore=E203,E302,E305,E703
nbqa pylint code_horizontal.ipynb --disable=C0114
nbqa ruff code_horizontal.ipynb
nbqa mypy code_horizontal.ipynb --ignore-missing-imports
nbqa pydocstyle code_horizontal.ipynb
