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

from flwr.common import Context, RecordSet
from flwr.common.logger import warn_preview_feature
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
        context = Context(state=RecordSet(), run_config={})
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
