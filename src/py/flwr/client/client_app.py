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
"""Flower ClientApp."""


import inspect
from typing import Callable, List, Optional

from flwr.client.client import Client
from flwr.client.message_handler.message_handler import (
    handle_legacy_message_from_msgtype,
)
from flwr.client.mod.utils import make_ffn
from flwr.client.typing import ClientFnExt, Mod
from flwr.common import Context, Message, MessageType
from flwr.common.logger import warn_deprecated_feature, warn_preview_feature

from .typing import ClientAppCallable


def _inspect_maybe_adapt_client_fn_signature(client_fn: ClientFnExt) -> ClientFnExt:
    client_fn_args = inspect.signature(client_fn).parameters
    first_arg = list(client_fn_args.keys())[0]

    if len(client_fn_args) != 1 or client_fn_args[first_arg].annotation is not Context:
        warn_deprecated_feature(
            "`client_fn` now expects a signature `def client_fn(context: Context)`."
            "\The provided `client_fn` has signature: "
            f"{dict(client_fn_args.items())}"
        )

        # Wrap depcreated client_fn inside a function with the expected signature
        def adaptor_fn(context: Context) -> Client:  # pylint: disable=unused-argument
            # if patition-id is defined, pass it. Else pass node_id that should always
            # be defined during Context init.
            cid = context.node_config.get("partition-id", context.node_id)
            return client_fn(str(cid))  # type: ignore

        return adaptor_fn

    return client_fn


class ClientAppException(Exception):
    """Exception raised when an exception is raised while executing a ClientApp."""

    def __init__(self, message: str):
        ex_name = self.__class__.__name__
        self.message = f"\nException {ex_name} occurred. Message: " + message
        super().__init__(self.message)


class ClientApp:
    """Flower ClientApp.

    Examples
    --------
    Assuming a typical `Client` implementation named `FlowerClient`, you can wrap it in
    a `ClientApp` as follows:

    >>> class FlowerClient(NumPyClient):
    >>>     # ...
    >>>
    >>> def client_fn(node_id: int, partition_id: Optional[int]):
    >>>    return FlowerClient().to_client()
    >>>
    >>> app = ClientApp(client_fn)

    If the above code is in a Python module called `client`, it can be started as
    follows:

    >>> flower-client-app client:app --insecure

    In this `client:app` example, `client` refers to the Python module `client.py` in
    which the previous code lives in and `app` refers to the global attribute `app` that
    points to an object of type `ClientApp`.
    """

    def __init__(
        self,
        client_fn: Optional[ClientFnExt] = None,  # Only for backward compatibility
        mods: Optional[List[Mod]] = None,
    ) -> None:
        self._mods: List[Mod] = mods if mods is not None else []

        # Create wrapper function for `handle`
        self._call: Optional[ClientAppCallable] = None
        if client_fn is not None:

            client_fn = _inspect_maybe_adapt_client_fn_signature(client_fn)

            def ffn(
                message: Message,
                context: Context,
            ) -> Message:  # pylint: disable=invalid-name
                out_message = handle_legacy_message_from_msgtype(
                    client_fn=client_fn, message=message, context=context
                )
                return out_message

            # Wrap mods around the wrapped handle function
            self._call = make_ffn(ffn, mods if mods is not None else [])

        # Step functions
        self._train: Optional[ClientAppCallable] = None
        self._evaluate: Optional[ClientAppCallable] = None
        self._query: Optional[ClientAppCallable] = None

    def __call__(self, message: Message, context: Context) -> Message:
        """Execute `ClientApp`."""
        # Execute message using `client_fn`
        if self._call:
            return self._call(message, context)

        # Execute message using a new
        if message.metadata.message_type == MessageType.TRAIN:
            if self._train:
                return self._train(message, context)
            raise ValueError("No `train` function registered")
        if message.metadata.message_type == MessageType.EVALUATE:
            if self._evaluate:
                return self._evaluate(message, context)
            raise ValueError("No `evaluate` function registered")
        if message.metadata.message_type == MessageType.QUERY:
            if self._query:
                return self._query(message, context)
            raise ValueError("No `query` function registered")

        # Message type did not match one of the known message types abvoe
        raise ValueError(f"Unknown message_type: {message.metadata.message_type}")

    def train(self) -> Callable[[ClientAppCallable], ClientAppCallable]:
        """Return a decorator that registers the train fn with the client app.

        Examples
        --------
        >>> app = ClientApp()
        >>>
        >>> @app.train()
        >>> def train(message: Message, context: Context) -> Message:
        >>>    print("ClientApp training running")
        >>>    # Create and return an echo reply message
        >>>    return message.create_reply(content=message.content())
        """

        def train_decorator(train_fn: ClientAppCallable) -> ClientAppCallable:
            """Register the train fn with the ServerApp object."""
            if self._call:
                raise _registration_error(MessageType.TRAIN)

            warn_preview_feature("ClientApp-register-train-function")

            # Register provided function with the ClientApp object
            # Wrap mods around the wrapped step function
            self._train = make_ffn(train_fn, self._mods)

            # Return provided function unmodified
            return train_fn

        return train_decorator

    def evaluate(self) -> Callable[[ClientAppCallable], ClientAppCallable]:
        """Return a decorator that registers the evaluate fn with the client app.

        Examples
        --------
        >>> app = ClientApp()
        >>>
        >>> @app.evaluate()
        >>> def evaluate(message: Message, context: Context) -> Message:
        >>>    print("ClientApp evaluation running")
        >>>    # Create and return an echo reply message
        >>>    return message.create_reply(content=message.content())
        """

        def evaluate_decorator(evaluate_fn: ClientAppCallable) -> ClientAppCallable:
            """Register the evaluate fn with the ServerApp object."""
            if self._call:
                raise _registration_error(MessageType.EVALUATE)

            warn_preview_feature("ClientApp-register-evaluate-function")

            # Register provided function with the ClientApp object
            # Wrap mods around the wrapped step function
            self._evaluate = make_ffn(evaluate_fn, self._mods)

            # Return provided function unmodified
            return evaluate_fn

        return evaluate_decorator

    def query(self) -> Callable[[ClientAppCallable], ClientAppCallable]:
        """Return a decorator that registers the query fn with the client app.

        Examples
        --------
        >>> app = ClientApp()
        >>>
        >>> @app.query()
        >>> def query(message: Message, context: Context) -> Message:
        >>>    print("ClientApp query running")
        >>>    # Create and return an echo reply message
        >>>    return message.create_reply(content=message.content())
        """

        def query_decorator(query_fn: ClientAppCallable) -> ClientAppCallable:
            """Register the query fn with the ServerApp object."""
            if self._call:
                raise _registration_error(MessageType.QUERY)

            warn_preview_feature("ClientApp-register-query-function")

            # Register provided function with the ClientApp object
            # Wrap mods around the wrapped step function
            self._query = make_ffn(query_fn, self._mods)

            # Return provided function unmodified
            return query_fn

        return query_decorator


class LoadClientAppError(Exception):
    """Error when trying to load `ClientApp`."""


def _registration_error(fn_name: str) -> ValueError:
    return ValueError(
        f"""Use either `@app.{fn_name}()` or `client_fn`, but not both.

        Use the `ClientApp` with an existing `client_fn`:

        >>> class FlowerClient(NumPyClient):
        >>>     # ...
        >>>
        >>> def client_fn(cid) -> Client:
        >>>     return FlowerClient().to_client()
        >>>
        >>> app = ClientApp(
        >>>     client_fn=client_fn,
        >>> )

        Use the `ClientApp` with a custom {fn_name} function:

        >>> app = ClientApp()
        >>>
        >>> @app.{fn_name}()
        >>> def {fn_name}(message: Message, context: Context) -> Message:
        >>>    print("ClientApp {fn_name} running")
        >>>    # Create and return an echo reply message
        >>>    return message.create_reply(
        >>>        content=message.content()
        >>>    )
        """,
    )
