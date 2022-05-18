# Copyright 2020 Adap GmbH. All Rights Reserved.
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
"""Flower client app."""


import time
from logging import INFO
from typing import Optional, Union

from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.logger import log

from .client import Client
from .grpc_client.connection import grpc_connection
from .grpc_client.message_handler import handle
from .numpy_client import NumPyClient, NumPyClientWrapper
from .numpy_client import has_get_properties as numpyclient_has_get_properties

ClientLike = Union[Client, NumPyClient]


def start_client(
    server_address: str,
    client: Client,
    grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    root_certificates: Optional[bytes] = None,
) -> None:
    """Start a Flower Client which connects to a gRPC server.

    Parameters
    ----------
        server_address: str. The IPv6 address of the server. If the Flower
            server runs on the same machine on port 8080, then `server_address`
            would be `"[::]:8080"`.
        client: flwr.client.Client. An implementation of the abstract base
            class `flwr.client.Client`.
        grpc_max_message_length: int (default: 536_870_912, this equals 512MB).
            The maximum length of gRPC messages that can be exchanged with the
            Flower server. The default should be sufficient for most models.
            Users who train very large models might need to increase this
            value. Note that the Flower server needs to be started with the
            same value (see `flwr.server.start_server`), otherwise it will not
            know about the increased limit and block larger messages.
        root_certificates: bytes (default: None)
            The PEM-encoded root certificates as a byte string. If provided, a secure
            connection using the certificates will be established to a
            SSL-enabled Flower server.

    Returns
    -------
        None

    Examples
    --------
    Starting a client with insecure server connection:

    >>> start_client(
    >>>     server_address=localhost:8080,
    >>>     client=FlowerClient(),
    >>> )

    Starting a SSL-enabled client:

    >>> from pathlib import Path
    >>> start_client(
    >>>     server_address=localhost:8080,
    >>>     client=FlowerClient(),
    >>>     root_certificates=Path("/crts/root.pem").read_bytes(),
    >>> )
    """
    while True:
        sleep_duration: int = 0
        with grpc_connection(
            server_address,
            max_message_length=grpc_max_message_length,
            root_certificates=root_certificates,
        ) as conn:
            receive, send = conn

            while True:
                server_message = receive()
                client_message, sleep_duration, keep_going = handle(
                    client, server_message
                )
                send(client_message)
                if not keep_going:
                    break
        if sleep_duration == 0:
            log(INFO, "Disconnect and shut down")
            break
        # Sleep and reconnect afterwards
        log(
            INFO,
            "Disconnect, then re-establish connection after %s second(s)",
            sleep_duration,
        )
        time.sleep(sleep_duration)


def start_numpy_client(
    server_address: str,
    client: NumPyClient,
    grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    root_certificates: Optional[bytes] = None,
) -> None:
    """Start a Flower NumPyClient which connects to a gRPC server.

    Parameters
    ----------
        server_address: str. The IPv6 address of the server. If the Flower
            server runs on the same machine on port 8080, then `server_address`
            would be `"[::]:8080"`.
        client: flwr.client.NumPyClient. An implementation of the abstract base
            class `flwr.client.NumPyClient`.
        grpc_max_message_length: int (default: 536_870_912, this equals 512MB).
            The maximum length of gRPC messages that can be exchanged with the
            Flower server. The default should be sufficient for most models.
            Users who train very large models might need to increase this
            value. Note that the Flower server needs to be started with the
            same value (see `flwr.server.start_server`), otherwise it will not
            know about the increased limit and block larger messages.
        root_certificates: bytes (default: None)
            The PEM-encoded root certificates a byte string. If provided, a secure
            connection using the certificates will be established to a
            SSL-enabled Flower server.

    Returns
    -------
        None

    Examples
    --------
    Starting a client with an insecure server connection:

    >>> start_client(
    >>>     server_address=localhost:8080,
    >>>     client=FlowerClient(),
    >>> )

    Starting a SSL-enabled client:

    >>> from pathlib import Path
    >>> start_client(
    >>>     server_address=localhost:8080,
    >>>     client=FlowerClient(),
    >>>     root_certificates=Path("/crts/root.pem").read_bytes(),
    >>> )
    """

    # Wrap the NumPyClient
    flower_client = NumPyClientWrapper(client)

    # Delete get_properties method from NumPyClientWrapper if the user-provided
    # NumPyClient instance does not implement get_properties. This enables the
    # following call to start_client to handle NumPyClientWrapper instances like any
    # other Client instance (which might or might not implement get_properties).
    if not numpyclient_has_get_properties(client=client):
        del NumPyClientWrapper.get_properties

    # Start
    start_client(
        server_address=server_address,
        client=flower_client,
        grpc_max_message_length=grpc_max_message_length,
        root_certificates=root_certificates,
    )


def to_client(client_like: ClientLike) -> Client:
    """Take any Client-like object and return it as a Client."""
    if isinstance(client_like, NumPyClient):
        return NumPyClientWrapper(numpy_client=client_like)
    return client_like
