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
"""Provides class ClientProxy."""

from queue import Queue

from flower.proto.transport_pb2 import ClientMessage, ServerMessage


class ClientProxy:
    """ClientProxy holding client_message and server_message."""

    def __init__(self) -> None:
        """Create message queues."""
        # Disable all unsubscriptable-object violations in __init__ method
        # pylint: disable=unsubscriptable-object
        self.client_message: Queue[ClientMessage] = Queue(maxsize=1)
        self.server_message: Queue[ServerMessage] = Queue(maxsize=1)

    def set_server_message_get_client_message(
        self, server_message: ServerMessage
    ) -> ClientMessage:
        """Set server message and return next client message."""
        self.server_message.put(server_message)
        return self.client_message.get()

    def set_client_message_get_server_message(
        self, client_message: ClientMessage
    ) -> ServerMessage:
        """Set client message and return next server message."""
        self.client_message.put(client_message)
        return self.server_message.get()
