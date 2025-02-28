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


from collections.abc import Iterator
from contextlib import contextmanager
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


@contextmanager
def _empty_lifecycle(_: Context) -> Iterator[None]:
    yield


class ServerApp:  # pylint: disable=too-many-instance-attributes
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
    >>>     )
    >>>
    >>> app = ServerApp(server_fn=server_fn)

    Use the `ServerApp` with a custom main function:

    >>> app = ServerApp()
    >>>
    >>> @app.main()
    >>> def main(driver: Driver, context: Context) -> None:
    >>>    print("ServerApp running")
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
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
        self._lifecycle = _empty_lifecycle

    def __call__(self, driver: Driver, context: Context) -> None:
        """Execute `ServerApp`."""
        with self._lifecycle(context):
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

    def lifecycle(
        self,
    ) -> Callable[
        [Callable[[Context], Iterator[None]]], Callable[[Context], Iterator[None]]
    ]:
        """Return a decorator that registers the lifecycle fn with the server app.

        The decorated function should accept a `Context` object and use `yield`
        to define enter and exit behavior.

        Examples
        --------
        >>> app = ServerApp()
        >>>
        >>> @app.lifecycle()
        >>> def lifecycle(context: Context) -> None:
        >>>     print("Initializing ServerApp")
        >>>     yield
        >>>     print("Cleaning up ServerApp")
        """

        def lifecycle_decorator(
            lifecycle_fn: Callable[[Context], Iterator[None]]
        ) -> Callable[[Context], Iterator[None]]:
            """Register the lifecycle fn with the ServerApp object."""
            warn_preview_feature("ServerApp-register-lifecycle-function")

            @contextmanager
            def decorated_lifecycle(context: Context) -> Iterator[None]:
                # Execute the code before `yield` in lifecycle_fn
                try:
                    if not isinstance(it := lifecycle_fn(context), Iterator):
                        raise StopIteration
                    next(it)
                except StopIteration:
                    raise RuntimeError(
                        "Lifecycle function should yield at least once."
                    ) from None

                try:
                    # Enter the context
                    yield
                finally:
                    try:
                        # Execute the code after `yield` in lifecycle_fn
                        next(it)
                    except StopIteration:
                        pass
                    else:
                        raise RuntimeError("Lifecycle function should only yield once.")

            # Register provided function with the ServerApp object
            # Ignore mypy error because of different argument names (`_` vs `context`)
            self._lifecycle = decorated_lifecycle  # type: ignore

            # Return provided function unmodified
            return lifecycle_fn

        return lifecycle_decorator


class LoadServerAppError(Exception):
    """Error when trying to load `ServerApp`."""
