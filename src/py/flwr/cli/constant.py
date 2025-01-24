# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Constants for CLI commands."""


# The help message for `--federation-config` option
FEDERATION_CONFIG_HELP_MESSAGE = (
    "Override federation configuration values in the format:\n\n"
    "`--federation-config 'key1=value1 key2=value2' --federation-config "
    "'key3=value3'`\n\nValues can be of any type supported in TOML, such as "
    "bool, int, float, or string. Ensure that the keys (`key1`, `key2`, `key3` "
    "in this example) exist in the federation configuration under the "
    "`[tool.flwr.federations.<YOUR_FEDERATION>]` table of the `pyproject.toml` "
    "for proper overriding."
)
