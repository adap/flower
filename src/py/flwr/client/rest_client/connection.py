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
"""Contextmanager managing a REST-based channel to the Flower server."""


from contextlib import contextmanager
from typing import Callable, Iterator, Optional, Tuple

import requests

from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.proto.transport_pb2 import ClientMessage, ServerMessage


@contextmanager
def rest_not_a_connection(
    client_id: str,
    server_address: str,
    max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    root_certificates: Optional[bytes] = None,
) -> Iterator[
    Tuple[Callable[[], Optional[ServerMessage]], Callable[[ClientMessage], None]]
]:
    """Primitives for request/response-based interaction with a server.

    One notable difference to the grpc_connection context manager is that
    `receive` can return `None`.

    Parameters
    ----------
    client_id: str
        TODO Ignored, only present to preserve API-compatibility.
    server_address : str
        The IPv6 address of the server. If the Flower server runs on the same machine
        on port 8080, then `server_address` would be `"[::]:8080"`.
    max_message_length : int
        Ignored, only present to preserve API-compatibility.
    root_certificates : Optional[bytes] (default: None)
        Ignored, for now. TODO: enable secure connections

    Returns
    -------
    receive, send : Callable, Callable
    """

    base_url = f"http://{server_address}"

    ###########################################################################
    # receive/send functions
    ###########################################################################

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

    ###########################################################################
    # Protocol:
    # 1. Announce client-HERE to the server
    # 2. TODO periodically send client-BEAT (not yet handled on the server)
    # 3. Repeatedly (implemented by the calling code):
    #    1. Receive `ServerMessage` (instruction) or `None` from the server
    #    2. If `None`, ask again (back to 1.), else process `ServerMessage`
    #    3. Send `ClientMessage` (result) back to the server
    # 4. Announce client-AWAY to say this client is no longer available
    ###########################################################################

    # client-HERE
    r = requests.post(f"{base_url}/client/{client_id}")
    print(f"[C-{client_id}] POST /client/{client_id}:", r.status_code, r.headers)

    # client-BEAT / TODO move to background thread
    r = requests.put(f"{base_url}/client/{client_id}")
    print(f"[C-{client_id}] PUT /client/{client_id}:", r.status_code, r.headers)

    # Calling code will use receive/send to receive instructions and send
    # results, usually repeatedly
    try:
        yield (receive, send)
    finally:
        # client-AWAY
        r = requests.delete(f"{base_url}/client/{client_id}")
        print(f"[C-{client_id}] DELETE /client/{client_id}:", r.status_code, r.headers)
