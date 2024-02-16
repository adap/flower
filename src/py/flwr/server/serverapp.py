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
"""Flower ServerApp."""


import importlib
from typing import Optional, cast

from flwr.common.context import Context
from flwr.server.driver.driver import Driver
from flwr.server.strategy import Strategy

from .client_manager import ClientManager
from .compat import start_driver
from .server import Server
from .server_config import ServerConfig


class ServerApp:
    """Flower ServerApp."""

    def __init__(
        self,
        server: Optional[Server] = None,
        config: Optional[ServerConfig] = None,
        strategy: Optional[Strategy] = None,
        client_manager: Optional[ClientManager] = None,
    ) -> None:
        self.server = server
        self.config = config
        self.strategy = strategy
        self.client_manager = client_manager

    def __call__(self, driver: Driver, context: Context) -> None:
        """Execute `ServerApp`."""
        # Compatibility mode
        start_driver(
            server=self.server,
            config=self.config,
            strategy=self.strategy,
            client_manager=self.client_manager,
            driver=driver,
        )


class LoadServerAppError(Exception):
    """Error when trying to load `ServerApp`."""


def load_server_app(module_attribute_str: str) -> ServerApp:
    """Load the `ServerApp` object specified in a module attribute string.

    The module/attribute string should have the form <module>:<attribute>. Valid
    examples include `server:app` and `project.package.module:wrapper.app`. It
    must refer to a module on the PYTHONPATH, the module needs to have the specified
    attribute, and the attribute must be of type `ServerApp`.
    """
    module_str, _, attributes_str = module_attribute_str.partition(":")
    if not module_str:
        raise LoadServerAppError(
            f"Missing module in {module_attribute_str}",
        ) from None
    if not attributes_str:
        raise LoadServerAppError(
            f"Missing attribute in {module_attribute_str}",
        ) from None

    # Load module
    try:
        module = importlib.import_module(module_str)
    except ModuleNotFoundError:
        raise LoadServerAppError(
            f"Unable to load module {module_str}",
        ) from None

    # Recursively load attribute
    attribute = module
    try:
        for attribute_str in attributes_str.split("."):
            attribute = getattr(attribute, attribute_str)
    except AttributeError:
        raise LoadServerAppError(
            f"Unable to load attribute {attributes_str} from module {module_str}",
        ) from None

    # Check type
    if not isinstance(attribute, ServerApp):
        raise LoadServerAppError(
            f"Attribute {attributes_str} is not of type {ServerApp}",
        ) from None

    return cast(ServerApp, attribute)
