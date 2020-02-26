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

from typing import Optional
from queue import Queue

from flower.proto.transport_pb2 import ClientMessage, ServerMessage


class GRPCBridgeClosedError(Exception):
    """Signifies the bridge is closed."""


class GRPCBridge:
    """GRPCBridge holding client_message and server_message."""

    def __init__(self) -> None:
        """Create message queues."""
        # Disable all unsubscriptable-object violations in __init__ method
        # pylint: disable=unsubscriptable-object
        self.closed = False

        self._client_message: Queue[ClientMessage] = Queue(maxsize=1)
        self._server_message: Queue[ServerMessage] = Queue(maxsize=1)

    def abc(self, server_msg: ServerMessage) -> ClientMessage:
        self._server_message.put(server_msg)
        client_msg = self._client_message.get()
        return client_msg
        





    ########################################################################
    ########################################################################
    def close(self):
        """Set self.closed to true."""
        self.closed = True
        # In case there are open get calls this will unblock them
        self._server_message.put_nowait(None)
        self._client_message.put_nowait(None)

    def set_server_message(self, server_message: ServerMessage) -> None:
        """Set server message."""
        if self.closed:
            raise GRPCBridgeClosedError()

        self._server_message.put(server_message)

    def set_client_message(self, client_message: ClientMessage) -> None:
        """Set client message."""
        if self.closed:
            raise GRPCBridgeClosedError()

        self._client_message.put(client_message)

    def get_server_message(self) -> Optional[ServerMessage]:
        """Get server message."""
        if self.closed:
            raise GRPCBridgeClosedError()

        message = self._server_message.get()

        if message is None:
            raise GRPCBridgeClosedError()

        return message

    def get_client_message(self) -> Optional[ClientMessage]:
        """Get client message."""
        if self.closed:
            raise GRPCBridgeClosedError()

        message = self._client_message.get()

        if message is None:
            raise GRPCBridgeClosedError()

        return message
