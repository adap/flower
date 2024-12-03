# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""Flower EE auth plugins."""

from flwr.common.auth_plugin import CliAuthPlugin, ExecAuthPlugin
from .keycloak_plugin import KeycloakCliPlugin
from .keycloak_plugin import KeycloakExecPlugin

def get_cli_auth_plugins() -> dict[str, type[CliAuthPlugin]]:
    return {
        "keycloak": KeycloakCliPlugin
    }


def get_exec_auth_plugins() -> dict[str, type[ExecAuthPlugin]]:
    return {
        "keycloak": KeycloakExecPlugin
    }


__all__ = [
    "get_cli_auth_plugins",
    "get_exec_auth_plugins",
]
