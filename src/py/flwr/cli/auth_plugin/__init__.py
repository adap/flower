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
"""Flower user auth plugins."""


from flwr.common.auth_plugin import CliAuthPlugin
from flwr.common.constant import AuthType

from .oidc_cli_plugin import OidcCliPlugin


def get_cli_auth_plugins() -> dict[str, type[CliAuthPlugin]]:
    """Return all CLI authentication plugins."""
    return {AuthType.OIDC: OidcCliPlugin}


__all__ = [
    "get_cli_auth_plugins",
]
