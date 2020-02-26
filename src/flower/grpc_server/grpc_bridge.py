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
"""Provides class GRPCBridge."""

from queue import Queue
from typing import Iterator

from flower.proto.transport_pb2 import ClientMessage, ServerMessage


class GRPCBridge:
    """GRPCBridge holding client_message and server_message."""

    def __init__(self) -> None:
        """Create message queues."""
        # Disable all unsubscriptable-object violations in __init__ method
        # pylint: disable=unsubscriptable-object
        self._client_message: Queue[ClientMessage] = Queue(maxsize=1)
        self._server_message: Queue[ServerMessage] = Queue(maxsize=1)

    def abc(self, server_msg: ServerMessage) -> ClientMessage:
        self._server_message.put(server_msg)
        client_msg = self._client_message.get()
        return client_msg

    def server_message_iterator(self) -> Iterator[ServerMessage]:
        return iter(self._server_message.get, None)

    def set_client_message(self, client_msg: ClientMessage) -> None:
        self._client_message.put(client_msg)

    def close(self) -> None:
        self._server_message.put(None)
        self._client_message.put(None)
