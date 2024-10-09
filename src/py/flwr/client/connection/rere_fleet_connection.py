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
"""Connection for a request-response channel to the SuperLink."""


from __future__ import annotations

import random
import threading
from abc import abstractmethod
from copy import copy
from logging import ERROR

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
from flwr.common.logger import log
from flwr.common.message import Message, Metadata
from flwr.common.retry_invoker import RetryInvoker
from flwr.common.serde import (
    message_from_taskins,
    message_to_taskres,
    user_config_from_proto,
)
from flwr.common.typing import Fab, Run
from flwr.proto.fab_pb2 import GetFabRequest, GetFabResponse  # pylint: disable=E0611
from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
    CreateNodeRequest,
    CreateNodeResponse,
    DeleteNodeRequest,
    PingRequest,
    PingResponse,
    PullTaskInsRequest,
    PushTaskResRequest,
)
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse  # pylint: disable=E0611
from flwr.proto.task_pb2 import TaskIns  # pylint: disable=E0611

from .fleet_api import FleetApi
from .fleet_connection import FleetConnection


class RereFleetConnection(FleetConnection):
    """Network-based request-response fleet connection."""

    def __init__(  # pylint: disable=R0913, R0914, R0915
        self,
        server_address: str,
        insecure: bool,
        retry_invoker: RetryInvoker,
        max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,  # pylint: disable=W0613
        root_certificates: bytes | str | None = None,
        authentication_keys: (
            tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey] | None
        ) = None,
    ) -> None:
        """Initialize the RereFleetConnection."""
        super().__init__(
            server_address=server_address,
            insecure=insecure,
            retry_invoker=retry_invoker,
            max_message_length=max_message_length,
            root_certificates=root_certificates,
            authentication_keys=authentication_keys,
        )
        self.msg_id_to_metadata: dict[str, Metadata] = {}
        self.node: Node | None = None
        self.ping_thread: threading.Thread | None = None
        self.ping_stop_event = threading.Event()
        self.channel: grpc.Channel | None = None

    @property
    @abstractmethod
    def api(self) -> FleetApi:
        """The proxy providing low-level access to the Fleet API server."""

    def ping(self) -> None:
        """Ping the SuperLink."""
        # Get Node
        if self.node is None:
            log(ERROR, "Node instance missing")
            return

        # Construct the ping request
        req = PingRequest(node=self.node, ping_interval=PING_DEFAULT_INTERVAL)

        # Call FleetAPI
        res: PingResponse = self.api.Ping(req, timeout=PING_CALL_TIMEOUT)

        # Check if success
        if not res.success:
            raise RuntimeError("Ping failed unexpectedly.")

        # Wait
        rd = random.uniform(*PING_RANDOM_RANGE)
        next_interval: float = PING_DEFAULT_INTERVAL - PING_CALL_TIMEOUT
        next_interval *= PING_BASE_MULTIPLIER + rd
        if not self.ping_stop_event.is_set():
            self.ping_stop_event.wait(next_interval)

    def create_node(self) -> int | None:
        """Request to create a node."""
        # Call FleetAPI
        req = CreateNodeRequest(ping_interval=PING_DEFAULT_INTERVAL)
        res: CreateNodeResponse = self.retrier.invoke(
            self.api.CreateNode,
            request=req,
        )

        # Remember the node and the ping-loop thread
        self.node = res.node
        self.ping_thread = start_ping_loop(self.ping, self.ping_stop_event)
        return self.node.node_id

    def delete_node(self) -> None:
        """Request to create a node."""
        # Get Node
        if self.node is None:
            log(ERROR, "Node instance missing")
            return

        # Stop the ping-loop thread
        self.ping_stop_event.set()

        # Call FleetAPI
        req = DeleteNodeRequest(node=self.node)
        self.retrier.invoke(self.api.DeleteNode, request=req)

        # Cleanup
        self.node = None

    def receive(self) -> Message | None:
        """Receive a message."""
        # Get Node
        if self.node is None:
            log(ERROR, "Node instance missing")
            return None

        # Request instructions (task) from server
        req = PullTaskInsRequest(node=self.node)
        res = self.retrier.invoke(self.api.PullTaskIns, request=req)

        # Get the current TaskIns
        task_ins: TaskIns | None = get_task_ins(res)

        # Discard the current TaskIns if not valid
        if task_ins is not None and not (
            task_ins.task.consumer.node_id == self.node.node_id
            and validate_task_ins(task_ins)
        ):
            task_ins = None

        # Construct the Message
        in_message = message_from_taskins(task_ins) if task_ins else None

        # Remember `metadata` of the in message
        if in_message:
            metadata = copy(in_message.metadata)
            self.msg_id_to_metadata[metadata.message_id] = metadata

        # Return the message if available
        return in_message

    def send(self, message: Message) -> None:
        """Send a message."""
        # Get Node
        if self.node is None:
            log(ERROR, "Node instance missing")
            return

        # Get the metadata of the incoming message
        metadata = self.msg_id_to_metadata.get(message.metadata.reply_to_message)
        if metadata is None:
            log(ERROR, "No current message")
            return

        # Validate out message
        if not validate_out_message(message, metadata):
            log(ERROR, "Invalid out message")
            return

        # Construct TaskRes
        task_res = message_to_taskres(message)

        # Serialize ProtoBuf to bytes
        req = PushTaskResRequest(node=self.node, task_res_list=[task_res])
        self.retrier.invoke(self.api.PushTaskRes, req)

        # Cleanup
        metadata = None

    def get_run(self, run_id: int) -> Run:
        """Get run info."""
        # Call FleetAPI
        req = GetRunRequest(node=self.node, run_id=run_id)
        res: GetRunResponse = self.retrier.invoke(
            self.api.GetRun,
            request=req,
        )

        # Return fab_id and fab_version
        return Run(
            run_id,
            res.run.fab_id,
            res.run.fab_version,
            res.run.fab_hash,
            user_config_from_proto(res.run.override_config),
        )

    def get_fab(self, fab_hash: str) -> Fab:
        """Get FAB file."""
        # Call FleetAPI
        req = GetFabRequest(node=self.node, hash_str=fab_hash)
        res: GetFabResponse = self.retrier.invoke(
            self.api.GetFab,
            request=req,
        )

        return Fab(res.fab.hash_str, res.fab.content)
