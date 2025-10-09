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
"""Account auth plugin for ControlServicer."""


from flwr.common.constant import AuthnType, AuthzType

from .auth_plugin import ControlAuthnPlugin, ControlAuthzPlugin
from .noop_auth_plugin import NoOpControlAuthnPlugin, NoOpControlAuthzPlugin

try:
    from flwr.ee import get_control_authn_ee_plugins, get_control_authz_ee_plugins
except ImportError:

    def get_control_authn_ee_plugins() -> dict[str, type[ControlAuthnPlugin]]:
        """Return all Control API authentication plugins for EE."""
        return {}

    def get_control_authz_ee_plugins() -> dict[str, type[ControlAuthzPlugin]]:
        """Return all Control API authorization plugins for EE."""
        return {}


def get_control_authn_plugins() -> dict[str, type[ControlAuthnPlugin]]:
    """Return all Control API authentication plugins."""
    ee_dict: dict[str, type[ControlAuthnPlugin]] = get_control_authn_ee_plugins()
    return ee_dict | {AuthnType.NOOP: NoOpControlAuthnPlugin}


def get_control_authz_plugins() -> dict[str, type[ControlAuthzPlugin]]:
    """Return all Control API authorization plugins."""
    ee_dict: dict[str, type[ControlAuthzPlugin]] = get_control_authz_ee_plugins()
    return ee_dict | {AuthzType.NOOP: NoOpControlAuthzPlugin}


__all__ = [
    "ControlAuthnPlugin",
    "ControlAuthzPlugin",
    "NoOpControlAuthnPlugin",
    "NoOpControlAuthzPlugin",
    "get_control_authn_plugins",
    "get_control_authz_plugins",
]
