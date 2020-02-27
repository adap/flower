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

from enum import Enum
from threading import Condition
from typing import Iterator, Optional

from flower.proto.transport_pb2 import ClientMessage, ServerMessage


class GRPCBridgeClosed(Exception):
    """Error signaling that GRPCBridge is closed."""


class Status(Enum):
    """Status through which the brige can transition."""

    AWAITING_SERVER_MESSAGE = 1
    SERVER_MESSAGE_AVAILABLE = 2
    AWAITING_CLIENT_MESSAGE = 3
    CLIENT_MESSAGE_AVAILABLE = 4
    CLOSED = 5


class GRPCBridge:
    git 

    def __init__(self) -> None:
        """Create message queues."""
        # Disable all unsubscriptable-object violations in __init__ method
        # pylint: disable=unsubscriptable-object
        self._cv = Condition()
        self._status = Status.AWAITING_SERVER_MESSAGE
        self._server_message: Optional[ServerMessage] = None
        self._client_message: Optional[ClientMessage] = None

    def _is_closed(self) -> bool:
        """Return True if closed and False otherwise."""
        return self._status == Status.CLOSED

    def _raise_if_closed(self) -> None:
        if self._status == Status.CLOSED:
            raise GRPCBridgeClosed()

    def _transition(self, next_status: Status) -> None:
        """Validates and transitions bridge."""
        with self._cv:
            if next_status == Status.CLOSED:
                self._status = next_status
            elif (
                self._status == Status.AWAITING_SERVER_MESSAGE
                and next_status == Status.SERVER_MESSAGE_AVAILABLE
            ):
                self._status = next_status
            elif (
                self._status == Status.SERVER_MESSAGE_AVAILABLE
                and next_status == Status.AWAITING_CLIENT_MESSAGE
            ):
                self._status = next_status
            elif (
                self._status == Status.AWAITING_CLIENT_MESSAGE
                and next_status == Status.CLIENT_MESSAGE_AVAILABLE
            ):
                self._status = next_status
            elif (
                self._status == Status.CLIENT_MESSAGE_AVAILABLE
                and next_status == Status.AWAITING_SERVER_MESSAGE
            ):
                self._status = next_status
            else:
                raise Exception(f"Invalid transition: {self._status} to {next_status}")

            self._cv.notify_all()

    def close(self) -> None:
        """Set bridge status to closed."""
        self._transition(Status.CLOSED)

    def request(self, server_message: ServerMessage) -> ClientMessage:
        """Set server massage and wait for client message."""
        print("request")
        # Set server message and transition to SERVER_MESSAGE_AVAILABLE
        with self._cv:
            self._cv.wait_for(
                lambda: self._status in [Status.CLOSED, Status.AWAITING_SERVER_MESSAGE]
            )
        self._raise_if_closed()
        self._server_message = server_message  # Write
        self._transition(Status.SERVER_MESSAGE_AVAILABLE)

        # Read client message and transition to AWAITING_SERVER_MESSAGE
        with self._cv:
            self._cv.wait_for(
                lambda: self._status in [Status.CLOSED, Status.CLIENT_MESSAGE_AVAILABLE]
            )
        self._raise_if_closed()
        client_message = self._client_message  # Read
        self._client_message = None  # Reset
        self._transition(Status.AWAITING_SERVER_MESSAGE)

        if client_message is None:
            raise Exception("Client message can not be None")

        return client_message

    def server_message_iterator(self) -> Iterator[ServerMessage]:
        """Return iterator over server messages."""
        print("next(server_message_iterator)")
        while not self._is_closed():
            with self._cv:
                self._cv.wait_for(
                    lambda: self._status
                    in [Status.CLOSED, Status.SERVER_MESSAGE_AVAILABLE]
                )
            self._raise_if_closed()
            server_message = self._server_message  # Read
            self._server_message = None  # Reset

            # Transition before yielding as after the yield the execution of this
            # function is paused and will resume when next is called again
            self._transition(Status.AWAITING_CLIENT_MESSAGE)

            if server_message is None:
                raise Exception("Server message can not be None")

            yield server_message

    def set_client_message(self, client_message: ClientMessage) -> None:
        """Set client message for consumption."""
        print("set_client_message")
        self._raise_if_closed()

        with self._cv:
            self._cv.wait_for(
                lambda: self._status in [Status.CLOSED, Status.AWAITING_CLIENT_MESSAGE]
            )
        self._raise_if_closed()
        self._client_message = client_message  # Write
        self._transition(Status.CLIENT_MESSAGE_AVAILABLE)


# class GRPCBridge:
#     """GRPCBridge holding client_message and server_message."""

#     def __init__(self) -> None:
#         """Create message queues."""
#         # Disable all unsubscriptable-object violations in __init__ method
#         # pylint: disable=unsubscriptable-object
#         self._client_message: Queue[ClientMessage] = Queue(maxsize=1)
#         self._server_message: Queue[ServerMessage] = Queue(maxsize=1)
#         self.closed = False

#     def request(self, server_message: ServerMessage) -> Optional[ClientMessage]:
#         """Set server massage and wait for client message."""
#         if self.closed:
#             return None

#         self._server_message.put(server_message)
#         client_message = self._client_message.get()

#         return client_message

#     def server_message_iterator(self) -> Iterator[ServerMessage]:
#         """Return iterator for _server_message queue."""
#         return iter(self._server_message.get, None)

#     def set_client_message(self, client_message: ClientMessage) -> None:
#         if not self.closed:
#             self._client_message.put(client_message)

#     def close(self) -> None:
#         """Put None into queues.
#         This will unblock get requests on the queues in case a request is still pending
#         and raise StopIteration for users of the server_message_iterator.
#         """
#         self.closed = True
#         try:
#             self._server_message.put_nowait(None)
#             self._client_message.put_nowait(None)
#         except Full:
#             pass
