# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""Flower SuperNode connection."""


from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from flwr.common import Message
from flwr.common.typing import Run
import random
import threading
from contextlib import contextmanager
from copy import copy
from logging import DEBUG, ERROR
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence, Tuple, Type, Union, cast

import grpc
from cryptography.hazmat.primitives.asymmetric import ec

from flwr.client.heartbeat import start_ping_loop
from flwr.client.message_handler.message_handler import validate_out_message
from flwr.client.message_handler.task_handler import get_task_ins, validate_task_ins
from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.constant import (
    PING_BASE_MULTIPLIER,
    PING_CALL_TIMEOUT,
    PING_DEFAULT_INTERVAL,
    PING_RANDOM_RANGE,
)
from flwr.common.grpc import create_channel
from flwr.common.logger import log
from flwr.common.message import Message, Metadata
from flwr.common.retry_invoker import RetryInvoker
from flwr.common.serde import (
    message_from_taskins,
    message_to_taskres,
    user_config_from_proto,
)
from flwr.common.typing import Fab, Run
from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
    CreateNodeRequest,
    DeleteNodeRequest,
    PingRequest,
    PingResponse,
    PullTaskInsRequest,
    PushTaskResRequest,
)
from flwr.proto.fleet_pb2_grpc import FleetStub  # pylint: disable=E0611
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse  # pylint: disable=E0611
from flwr.proto.task_pb2 import TaskIns  # pylint: disable=E0611

from .client_interceptor import AuthenticateClientInterceptor


class Connection(ABC):
    """Abstract base class for SuperNode connections."""
    
    @classmethod
    def open(  # pylint: disable=R0913, R0914, R0915
        cls,
        server_address: str,
        insecure: bool,
        retry_invoker: RetryInvoker,
        max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,  # pylint: disable=W0613
        root_certificates: Optional[Union[bytes, str]] = None,
        authentication_keys: Optional[
            Tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]
        ] = None,
    ) -> Connection:
        """Open a connection with the SuperLink.

        Parameters
        ----------
        server_address : str
            The IPv6 address of the server with `http://` or `https://`.
            If the Flower server runs on the same machine
            on port 8080, then `server_address` would be `"http://[::]:8080"`.
        insecure : bool
            Starts an insecure gRPC connection when True. Enables HTTPS connection
            when False, using system certificates if `root_certificates` is None.
        retry_invoker: RetryInvoker
            `RetryInvoker` object that will try to reconnect the client to the server
            after gRPC errors. If None, the client will only try to
            reconnect once after a failure.
        max_message_length : int
            Ignored, only present to preserve API-compatibility.
        root_certificates : Optional[Union[bytes, str]] (default: None)
            Path of the root certificate. If provided, a secure
            connection using the certificates will be established to an SSL-enabled
            Flower server. Bytes won't work for the REST API.
        authentication_keys : Optional[Tuple[PrivateKey, PublicKey]] (default: None)
            Tuple containing the elliptic curve private key and public key for
            authentication from the cryptography library.
            Source: https://cryptography.io/en/latest/hazmat/primitives/asymmetric/ec/
            Used to establish an authenticated connection with the server.
        """
        return cls(
            server_address=server_address,
            insecure=insecure,
            retry_invoker=retry_invoker,
            max_message_length=max_message_length,
            root_certificates=root_certificates,
            authentication_keys=authentication_keys,
        )

    @abstractmethod
    def create_node() -> Optional[int]:
        """Create node."""

    @abstractmethod
    def delete_node() -> None:
        """Delete node."""

    @abstractmethod
    def receive() -> Optional[Message]:
        """Receive message."""

    @abstractmethod
    def send(message: Message) -> None: 
        """Send message."""

    @abstractmethod
    def get_run(run_id: int) -> Run: 
        """Get run info."""
