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
from typing import Optional
from uuid import uuid4
from xmlrpc.client import Server

import requests

from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.logger import log
from flwr.proto.transport_pb2 import ClientMessage, ServerMessage

from .client import Client
from .grpc_client.connection import grpc_connection
from .grpc_client.message_handler import handle
from .numpy_client import NumPyClient, NumPyClientWrapper
from .numpy_client import has_get_properties as numpyclient_has_get_properties


def conn(base_url: str, client_id: str):
    """Provide receive/send primitives, similar to the way the gRPC works.

    One notable difference is that `receive` can return `None`.
    """

    def receive() -> Optional[ServerMessage]:
        """Receive next task from server."""

        # Request instructions (task) from server
        r = requests.get(f"{base_url}/ins/{client_id}")
        print(f"[C-{client_id}] GET /ins/{client_id}:", r.status_code, r.headers)
        if r.status_code != 200:
            return None

        # Deserialize ProtoBuf from bytes
        server_msg = ServerMessage()
        server_msg.ParseFromString(r.content)
        return server_msg

    def send(client_message: ClientMessage) -> None:
        """Send task result back to server."""

        # Serialize ProtoBuf to bytes
        client_msg_bytes = client_message.SerializeToString()

        # Send ClientMessage to server
        r = requests.post(
            f"{base_url}/res/{client_id}",
            headers={"Content-Type": "application/protobuf"},
            data=client_msg_bytes,
        )
        print(f"[C-{client_id}] POST /res/{client_id}:", r.status_code, r.headers)

    return receive, send


def start_rest_client(
    server_address: str,
    client: Client,
    grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    root_certificates: Optional[bytes] = None,
) -> None:
    """Start a Flower Client which connects to a REST server."""

    base_url = f"http://{server_address}"

    # Generate random client id (changes on every call to this function)
    client_id = str(uuid4().hex[:8])

    # Note: this loop is almost an exact copy of the loop in start_client. If
    # we wrapped the `requests` calls in a context manager then we could merge
    # both implementations into one.
    while True:
        sleep_duration: int = 0

        # client-HERE
        r = requests.post(f"{base_url}/client/{client_id}")
        print(f"[C-{client_id}] POST /client/{client_id}:", r.status_code, r.headers)

        # client-BEAT / TODO move to background thread
        r = requests.put(f"{base_url}/client/{client_id}")
        print(f"[C-{client_id}] PUT /client/{client_id}:", r.status_code, r.headers)

        receive, send = conn(base_url=base_url, client_id=client_id)

        while True:
            server_message = receive()
            if server_message is None:
                # TODO this should be pace-steered by the server
                time.sleep(2)  # Wait for 2s before asking again
                continue
            client_message, sleep_duration, keep_going = handle(client, server_message)
            send(client_message)
            if not keep_going:
                break

        # client-AWAY
        r = requests.delete(f"{base_url}/client/{client_id}")

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
    # start_client(
    #     server_address=server_address,
    #     client=flower_client,
    #     grpc_max_message_length=grpc_max_message_length,
    #     root_certificates=root_certificates,
    # )
    start_rest_client(
        server_address=server_address,
        client=flower_client,
        grpc_max_message_length=grpc_max_message_length,
        root_certificates=root_certificates,
    )
