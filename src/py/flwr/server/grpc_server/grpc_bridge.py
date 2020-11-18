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

from flwr.proto.transport_pb2 import ClientMessage, ServerMessage


class GRPCBridgeClosed(Exception):
    """Error signaling that GRPCBridge is closed."""


class Status(Enum):
    """Status through which the bridge can transition."""

    AWAITING_SERVER_MESSAGE = 1
    SERVER_MESSAGE_AVAILABLE = 2
    AWAITING_CLIENT_MESSAGE = 3
    CLIENT_MESSAGE_AVAILABLE = 4
    CLOSED = 5


class GRPCBridge:
    """GRPCBridge holding client_message and server_message.

    For understanding this class it is recommended to understand how
    the threading.Condition class works. See here:
    - https://docs.python.org/3/library/threading.html#condition-objects
    """

    def __init__(self) -> None:
        """Init bridge."""
        # Disable all unsubscriptable-object violations in __init__ method
        # pylint: disable=unsubscriptable-object
        self._cv = Condition()  # cv stands for condition variable
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
        """Validate status transition and set next status.

        The caller of the transition method will have to aquire
        conditional variable.
        """
        if next_status == Status.CLOSED:
            self._status = next_status
        elif (
            self._status == Status.AWAITING_SERVER_MESSAGE
            and next_status == Status.SERVER_MESSAGE_AVAILABLE
            and self._server_message is not None
            and self._client_message is None
        ):
            self._status = next_status
        elif (
            self._status == Status.SERVER_MESSAGE_AVAILABLE
            and next_status == Status.AWAITING_CLIENT_MESSAGE
            and self._server_message is None
            and self._client_message is None
        ):
            self._status = next_status
        elif (
            self._status == Status.AWAITING_CLIENT_MESSAGE
            and next_status == Status.CLIENT_MESSAGE_AVAILABLE
            and self._server_message is None
            and self._client_message is not None
        ):
            self._status = next_status
        elif (
            self._status == Status.CLIENT_MESSAGE_AVAILABLE
            and next_status == Status.AWAITING_SERVER_MESSAGE
            and self._server_message is None
            and self._client_message is None
        ):
            self._status = next_status
        else:
            raise Exception(f"Invalid transition: {self._status} to {next_status}")

        self._cv.notify_all()

    def close(self) -> None:
        """Set bridge status to closed."""
        with self._cv:
            self._transition(Status.CLOSED)

    def request(self, server_message: ServerMessage) -> ClientMessage:
        """Set server massage and wait for client message."""
        # Set server message and transition to SERVER_MESSAGE_AVAILABLE
        with self._cv:
            self._raise_if_closed()

            if self._status != Status.AWAITING_SERVER_MESSAGE:
                raise Exception("This should not happen")

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
                # function is paused and will resume when next is called again.
                # Also release condition variable by exiting the context
                self._transition(Status.AWAITING_CLIENT_MESSAGE)

            if server_message is None:
                raise Exception("Server message can not be None")

            yield server_message

    def set_client_message(self, client_message: ClientMessage) -> None:
        """Set client message for consumption."""
        with self._cv:
            self._raise_if_closed()

            if self._status != Status.AWAITING_CLIENT_MESSAGE:
                raise Exception("This should not happen")

            self._client_message = client_message  # Write
            self._transition(Status.CLIENT_MESSAGE_AVAILABLE)
