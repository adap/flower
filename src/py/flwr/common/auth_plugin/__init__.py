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
"""Auth plugin components."""


from .auth_plugin import ExecAuthPlugin as ExecAuthPlugin
from .auth_plugin import Metadata as Metadata
from .auth_plugin import UserAuthPlugin as UserAuthPlugin
from .keycloak_plugin import KeycloakExecPlugin as KeycloakExecPlugin
from .keycloak_plugin import KeycloakUserPlugin as KeycloakUserPlugin

__all__ = [
    "ExecAuthPlugin",
    "KeycloakExecPlugin",
    "KeycloakUserPlugin",
    "Metadata",
    "UserAuthPlugin",
]
