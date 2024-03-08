# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
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


from typing import List, Optional

from flwr.client.message_handler.message_handler import (
    handle_legacy_message_from_msgtype,
)
from flwr.client.mod.utils import make_ffn
from flwr.client.typing import ClientFn, Mod
from flwr.common import Context, Message


class ClientApp:
    """Flower ClientApp.

    Examples
    --------
    Assuming a typical `Client` implementation named `FlowerClient`, you can wrap it in
    a `ClientApp` as follows:

    >>> class FlowerClient(NumPyClient):
    >>>     # ...
    >>>
    >>> def client_fn(cid):
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
        client_fn: ClientFn,  # Only for backward compatibility
        mods: Optional[List[Mod]] = None,
    ) -> None:
        # Create wrapper function for `handle`
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

    def __call__(self, message: Message, context: Context) -> Message:
        """Execute `ClientApp`."""
        return self._call(message, context)


class LoadClientAppError(Exception):
    """Error when trying to load `ClientApp`."""
