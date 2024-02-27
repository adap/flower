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
from typing import Callable, Optional, cast

from flwr.common import Context, RecordSet
from flwr.server.strategy import Strategy

from .client_manager import ClientManager
from .compat import start_driver
from .driver import Driver
from .server import Server
from .server_config import ServerConfig
from .typing import ServerAppCallable


class ServerApp:
    """Flower ServerApp.

    Examples
    --------
    Use the `ServerApp` with an existing `Strategy`:

    >>> server_config = ServerConfig(num_rounds=3)
    >>> strategy = FedAvg()
    >>>
    >>> app = ServerApp()
    >>>     server_config=server_config,
    >>>     strategy=strategy,
    >>> )

    Use the `ServerApp` with a custom main function:

    >>> app = ServerApp()
    >>>
    >>> @app.main()
    >>> def main(driver: Driver, context: Context) -> None:
    >>>    print("ServerApp running")
    """

    def __init__(
        self,
        server: Optional[Server] = None,
        config: Optional[ServerConfig] = None,
        strategy: Optional[Strategy] = None,
        client_manager: Optional[ClientManager] = None,
    ) -> None:
        self._server = server
        self._config = config
        self._strategy = strategy
        self._client_manager = client_manager
        self._main: Optional[ServerAppCallable] = None

    def __call__(self, driver: Driver, context: Context) -> None:
        """Execute `ServerApp`."""
        # Compatibility mode
        if not self._main:
            start_driver(
                server=self._server,
                config=self._config,
                strategy=self._strategy,
                client_manager=self._client_manager,
                driver=driver,
            )
            return

        # New execution mode
        context = Context(state=RecordSet())
        self._main(driver, context)

    def main(self) -> Callable[[ServerAppCallable], ServerAppCallable]:
        """Return a decorator that registers the main fn with the server app.

        Examples
        --------
        >>> app = ServerApp()
        >>>
        >>> @app.main()
        >>> def main(driver: Driver, context: Context) -> None:
        >>>    print("ServerApp running")
        """

        def main_decorator(main_fn: ServerAppCallable) -> ServerAppCallable:
            """Register the main fn with the ServerApp object."""
            if self._server or self._config or self._strategy or self._client_manager:
                raise ValueError(
                    """Use either a custom main function or a `Strategy`, but not both.

                    Use the `ServerApp` with an existing `Strategy`:

                    >>> server_config = ServerConfig(num_rounds=3)
                    >>> strategy = FedAvg()
                    >>>
                    >>> app = ServerApp()
                    >>>     server_config=server_config,
                    >>>     strategy=strategy,
                    >>> )

                    Use the `ServerApp` with a custom main function:

                    >>> app = ServerApp()
                    >>>
                    >>> @app.main()
                    >>> def main(driver: Driver, context: Context) -> None:
                    >>>    print("ServerApp running")
                    """,
                )

            # Register provided function with the ServerApp object
            self._main = main_fn

            # Return provided function unmodified
            return main_fn

        return main_decorator


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
