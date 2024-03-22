# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
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
"""Provides class GrpcBridge."""


from dataclasses import dataclass
from enum import Enum
from threading import Condition, Lock
from typing import Iterator, Optional

from flwr.proto.transport_pb2 import (  # pylint: disable=E0611
    ClientMessage,
    ServerMessage,
)

from flwr.common import serde


@dataclass
class InsWrapper:
    """Instruction wrapper class for a single server message."""

    server_message: ServerMessage
    timeout: Optional[float]

    @property
    def is_end(self):
        return serde.is_server_message_end(self.server_message)


@dataclass
class ResWrapper:
    """Result wrapper class for a single client message."""

    client_message: ClientMessage

    
    @property
    def is_end(self):
        return serde.is_client_message_end(self.client_message)


class GrpcBridgeClosed(Exception):
    """Error signaling that GrpcBridge is closed."""


class Status(Enum):
    """Status through which the bridge can transition."""

    AWAITING_INS_WRAPPER = 1
    INS_WRAPPER_AVAILABLE_NO_TRANSITION = 2
    INS_WRAPPER_AVAILABLE_TRANSITION = 3
    AWAITING_RES_WRAPPER = 4
    RES_WRAPPER_AVAILABLE_NO_TRANSITION = 5
    RES_WRAPPER_AVAILABLE_TRANSITION = 6
    CLOSED = 7


class GrpcBridge:
    """GrpcBridge holding res_wrapper and ins_wrapper.

    For understanding this class it is recommended to understand how
    the threading.Condition class works. See here:
    - https://docs.python.org/3/library/threading.html#condition-objects
    """

    def __init__(self) -> None:
        """Init bridge."""
        # Disable all unsubscriptable-object violations in __init__ method
        # pylint: disable=unsubscriptable-object
        self._cv = Condition()  # cv stands for condition variable
        self._status = Status.AWAITING_INS_WRAPPER
        self._ins_wrapper: Optional[InsWrapper] = None
        self._res_wrapper: Optional[ResWrapper] = None
        self.ins_lock = Lock()

    def _is_closed(self) -> bool:
        """Return True if closed and False otherwise."""
        return self._status == Status.CLOSED

    def _raise_if_closed(self) -> None:
        if self._status == Status.CLOSED:
            raise GrpcBridgeClosed()

    def _transition(self, next_status: Status) -> None:
        """Validate status transition and set next status.

        The caller of the transition method will have to aquire conditional variable.
        """
        self._status = next_status
        self._cv.notify_all()

    def close(self) -> None:
        """Set bridge status to closed."""
        with self._cv:
            self._transition(Status.CLOSED)
    

    def read_res_wrapper(self) -> Iterator[ResWrapper]:
        while True:
            # Read res_wrapper and transition to AWAITING_INS_WRAPPER
            with self._cv:
                self._cv.wait_for(
                    lambda: self._status in [
                        Status.CLOSED,
                        Status.RES_WRAPPER_AVAILABLE_NO_TRANSITION,
                        Status.RES_WRAPPER_AVAILABLE_TRANSITION
                        ]
                )

                self._raise_if_closed()
                res_wrapper = self._res_wrapper  # Read
                self._res_wrapper = None  # Reset

                if self._status == Status.RES_WRAPPER_AVAILABLE_NO_TRANSITION:
                    self._transition(Status.AWAITING_RES_WRAPPER)
                elif self._status == Status.RES_WRAPPER_AVAILABLE_TRANSITION:
                    self._transition(Status.AWAITING_INS_WRAPPER) 

            if res_wrapper is None:
                raise ValueError("ResWrapper can not be None")

            yield res_wrapper
            if res_wrapper.is_end:
                break

    def write_ins_wrapper(self, ins_wrapper: InsWrapper, transition: bool):
        # Set ins_wrapper and transition to INS_WRAPPER_AVAILABLE
        with self._cv:
            self._raise_if_closed()
            
            self._cv.wait_for(
                lambda: self._status
                in [Status.AWAITING_INS_WRAPPER]
            )


            self._ins_wrapper = ins_wrapper  # Write
            if transition:
                self._transition(Status.INS_WRAPPER_AVAILABLE_TRANSITION)
            else:
                self._transition(Status.INS_WRAPPER_AVAILABLE_NO_TRANSITION)



    def request(self, ins_wrapper: InsWrapper | Iterator[InsWrapper]) -> Iterator[ResWrapper]:
        """Set ins_wrapper and wait for res_wrapper."""
        self.ins_lock.acquire()
        if isinstance(ins_wrapper, InsWrapper):
            self.write_ins_wrapper(ins_wrapper, True)
        else:
            wrappers = list(ins_wrapper)
            for i in range(len(wrappers)):
                wrapper = wrappers[i]
                self.write_ins_wrapper(wrapper, i == len(wrappers) - 1)
        self.ins_lock.release()

        yield from self.read_res_wrapper()


    def ins_wrapper_iterator(self) -> Iterator[InsWrapper]:
        """Return iterator over ins_wrapper objects."""
        while not self._is_closed():
            with self._cv:
                self._cv.wait_for(
                    lambda: self._status
                    in [Status.CLOSED,
                        Status.INS_WRAPPER_AVAILABLE_NO_TRANSITION,
                        Status.INS_WRAPPER_AVAILABLE_TRANSITION
                    ]
                )


                self._raise_if_closed()

                ins_wrapper = self._ins_wrapper  # Read
                self._ins_wrapper = None  # Reset

                # Transition before yielding as after the yield the execution of this
                # function is paused and will resume when next is called again.
                # Also release condition variable by exiting the context
                if self._status == Status.INS_WRAPPER_AVAILABLE_NO_TRANSITION:
                    self._transition(Status.AWAITING_INS_WRAPPER)
                else:
                    self._transition(Status.AWAITING_RES_WRAPPER)

            if ins_wrapper is None:
                raise ValueError("InsWrapper can not be None")

            yield ins_wrapper

    def set_res_wrapper(self, res_wrapper: ResWrapper, transition: bool) -> None:
        """Set res_wrapper for consumption."""
        with self._cv:
            self._raise_if_closed()
            self._cv.wait_for(
                    lambda: self._status
                    in [Status.AWAITING_RES_WRAPPER]
                )


            self._res_wrapper = res_wrapper  # Write
            if transition:
                self._transition(Status.RES_WRAPPER_AVAILABLE_TRANSITION)
            else:
                self._transition(Status.RES_WRAPPER_AVAILABLE_NO_TRANSITION)
