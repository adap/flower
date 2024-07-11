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


from typing import Callable, Optional

from flwr.common import Context
from flwr.common.logger import (
    warn_deprecated_feature_with_example,
    warn_preview_feature,
)
from flwr.server.strategy import Strategy

from .client_manager import ClientManager
from .compat import start_driver
from .driver import Driver
from .server import Server
from .server_config import ServerConfig
from .typing import ServerAppCallable, ServerFn

SERVER_FN_USAGE_EXAMPLE = """

        def server_fn(context: Context):
            server_config = ServerConfig(num_rounds=3)
            strategy = FedAvg()
            return ServerAppComponents(
                strategy=strategy,
                server_config=server_config,
        )

        app = ServerApp(server_fn=server_fn)
"""


class ServerApp:
    """Flower ServerApp.

    Examples
    --------
    Use the `ServerApp` with an existing `Strategy`:

    >>> def server_fn(context: Context):
    >>>     server_config = ServerConfig(num_rounds=3)
    >>>     strategy = FedAvg()
    >>>     return ServerAppComponents(
    >>>         strategy=strategy,
    >>>         server_config=server_config,
    >>>    )
    >>>
    >>> app = ServerApp(server_fn=server_fn)

    Use the `ServerApp` with a custom main function:

    >>> app = ServerApp()
    >>>
    >>> @app.main()
    >>> def main(driver: Driver, context: Context) -> None:
    >>>    print("ServerApp running")
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        server: Optional[Server] = None,
        config: Optional[ServerConfig] = None,
        strategy: Optional[Strategy] = None,
        client_manager: Optional[ClientManager] = None,
        server_fn: Optional[ServerFn] = None,
    ) -> None:
        if any([server, config, strategy, client_manager]):
            warn_deprecated_feature_with_example(
                deprecation_message="Passing either `server`, `config`, `strategy` or "
                "`client_manager` directly to the ServerApp "
                "constructor is deprecated.",
                example_message="Pass `ServerApp` arguments wrapped "
                "in a `flwr.server.ServerAppComponents` object that gets "
                "returned by a function passed as the `server_fn` argument "
                "to the `ServerApp` constructor. For example: ",
                code_example=SERVER_FN_USAGE_EXAMPLE,
            )

            if server_fn:
                raise ValueError(
                    "Passing `server_fn` is incompatible with passing the "
                    "other arguments (now deprecated) to ServerApp. "
                    "Use `server_fn` exclusively."
                )

        self._server = server
        self._config = config
        self._strategy = strategy
        self._client_manager = client_manager
        self._server_fn = server_fn
        self._main: Optional[ServerAppCallable] = None

    def __call__(self, driver: Driver, context: Context) -> None:
        """Execute `ServerApp`."""
        # Compatibility mode
        if not self._main:
            if self._server_fn:
                # Execute server_fn()
                components = self._server_fn(context)
                self._server = components.server
                self._config = components.config
                self._strategy = components.strategy
                self._client_manager = components.client_manager
            start_driver(
                server=self._server,
                config=self._config,
                strategy=self._strategy,
                client_manager=self._client_manager,
                driver=driver,
            )
            return

        # New execution mode
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
                    >>> app = ServerApp(
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

            warn_preview_feature("ServerApp-register-main-function")

            # Register provided function with the ServerApp object
            self._main = main_fn

            # Return provided function unmodified
            return main_fn

        return main_decorator


class LoadServerAppError(Exception):
    """Error when trying to load `ServerApp`."""
