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
"""Flower account auth plugins."""


from flwr.common.constant import AuthnType

from .auth_plugin import CliAuthPlugin, LoginError
from .noop_auth_plugin import NoOpCliAuthPlugin
from .oidc_cli_plugin import OidcCliPlugin


def get_cli_plugin_class(authn_type: str) -> type[CliAuthPlugin]:
    """Return all CLI authentication plugins."""
    if authn_type == AuthnType.NOOP:
        return NoOpCliAuthPlugin
    if authn_type == AuthnType.OIDC:
        return OidcCliPlugin
    raise ValueError(f"Unsupported authentication type: {authn_type}")


__all__ = [
    "CliAuthPlugin",
    "LoginError",
    "NoOpCliAuthPlugin",
    "OidcCliPlugin",
    "get_cli_plugin_class",
]
