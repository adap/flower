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


# General help message for config overrides
CONFIG_HELP_MESSAGE = (
    "Override {0} values using one of the following formats:\n\n"
    "--{1} '<k1>=<v1> <k2>=<v2>' | --{1} '<k1>=<v1>' --{1} '<k2>=<v2>'{2}\n\n"
    "When providing key-value pairs, values can be of any type supported by TOML "
    "(e.g., bool, int, float, string). The specified keys (<k1> and <k2> in the "
    "example) must exist in the {0} under the `{3}` section of `pyproject.toml` to be "
    "overridden.{4}"
)

# The help message for `--run-config` option
RUN_CONFIG_HELP_MESSAGE = CONFIG_HELP_MESSAGE.format(
    "run configuration",
    "run-config",
    " | --run-config <path/to/your/toml>",
    "[tool.flwr.app.config]",
    " Alternatively, provide a TOML file containing overrides.",
)

# The help message for `--federation-config` option
FEDERATION_CONFIG_HELP_MESSAGE = CONFIG_HELP_MESSAGE.format(
    "federation configuration",
    "federation-config",
    "",
    "[tool.flwr.federations.<YOUR-FEDERATION>]",
    "",
)
