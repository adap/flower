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
"""Flower gRPC Driver."""

import time
import warnings
from logging import DEBUG, ERROR, WARNING
from typing import Iterable, List, Optional, Tuple, cast

import grpc

from flwr.common import DEFAULT_TTL, EventType, Message, Metadata, RecordSet, event
from flwr.common.grpc import create_channel
from flwr.common.logger import log
from flwr.common.serde import message_from_taskres, message_to_taskins
from flwr.common.typing import Run
from flwr.proto.driver_pb2 import (  # pylint: disable=E0611
    CreateRunRequest,
    CreateRunResponse,
    GetNodesRequest,
    GetNodesResponse,
    PullTaskResRequest,
    PullTaskResResponse,
    PushTaskInsRequest,
    PushTaskInsResponse,
)
from flwr.proto.driver_pb2_grpc import DriverStub  # pylint: disable=E0611
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse  # pylint: disable=E0611
from flwr.proto.task_pb2 import TaskIns  # pylint: disable=E0611

from .driver import Driver

DEFAULT_SERVER_ADDRESS_DRIVER = "[::]:9091"

ERROR_MESSAGE_DRIVER_NOT_CONNECTED = """
[Driver] Error: Not connected.

Call `connect()` on the `GrpcDriverStub` instance before calling any of the other
`GrpcDriverStub` methods.
"""


class GrpcDriverStub:
    """`GrpcDriverStub` provides access to the gRPC Driver API/service.

    Parameters
    ----------
    driver_service_address : Optional[str]
        The IPv4 or IPv6 address of the Driver API server.
        Defaults to `"[::]:9091"`.
    root_certificates : Optional[bytes] (default: None)
        The PEM-encoded root certificates as a byte string.
        If provided, a secure connection using the certificates will be
        established to an SSL-enabled Flower server.
    """

    def __init__(
        self,
        driver_service_address: str = DEFAULT_SERVER_ADDRESS_DRIVER,
        root_certificates: Optional[bytes] = None,
    ) -> None:
        self.driver_service_address = driver_service_address
        self.root_certificates = root_certificates
        self.channel: Optional[grpc.Channel] = None
        self.stub: Optional[DriverStub] = None

    def is_connected(self) -> bool:
        """Return True if connected to the Driver API server, otherwise False."""
        return self.channel is not None

    def connect(self) -> None:
        """Connect to the Driver API."""
        event(EventType.DRIVER_CONNECT)
        if self.channel is not None or self.stub is not None:
            log(WARNING, "Already connected")
            return
        self.channel = create_channel(
            server_address=self.driver_service_address,
            insecure=(self.root_certificates is None),
            root_certificates=self.root_certificates,
        )
        self.stub = DriverStub(self.channel)
        log(DEBUG, "[Driver] Connected to %s", self.driver_service_address)

    def disconnect(self) -> None:
        """Disconnect from the Driver API."""
        event(EventType.DRIVER_DISCONNECT)
        if self.channel is None or self.stub is None:
            log(DEBUG, "Already disconnected")
            return
        channel = self.channel
        self.channel = None
        self.stub = None
        channel.close()
        log(DEBUG, "[Driver] Disconnected")

    def create_run(self, req: CreateRunRequest) -> CreateRunResponse:
        """Request for run ID."""
        # Check if channel is open
        if self.stub is None:
            log(ERROR, ERROR_MESSAGE_DRIVER_NOT_CONNECTED)
            raise ConnectionError("`GrpcDriverStub` instance not connected")

        # Call Driver API
        res: CreateRunResponse = self.stub.CreateRun(request=req)
        return res

    def get_run(self, req: GetRunRequest) -> GetRunResponse:
        """Get run information."""
        # Check if channel is open
        if self.stub is None:
            log(ERROR, ERROR_MESSAGE_DRIVER_NOT_CONNECTED)
            raise ConnectionError("`GrpcDriverStub` instance not connected")

        # Call gRPC Driver API
        res: GetRunResponse = self.stub.GetRun(request=req)
        return res

    def get_nodes(self, req: GetNodesRequest) -> GetNodesResponse:
        """Get client IDs."""
        # Check if channel is open
        if self.stub is None:
            log(ERROR, ERROR_MESSAGE_DRIVER_NOT_CONNECTED)
            raise ConnectionError("`GrpcDriverStub` instance not connected")

        # Call gRPC Driver API
        res: GetNodesResponse = self.stub.GetNodes(request=req)
        return res

    def push_task_ins(self, req: PushTaskInsRequest) -> PushTaskInsResponse:
        """Schedule tasks."""
        # Check if channel is open
        if self.stub is None:
            log(ERROR, ERROR_MESSAGE_DRIVER_NOT_CONNECTED)
            raise ConnectionError("`GrpcDriverStub` instance not connected")

        # Call gRPC Driver API
        res: PushTaskInsResponse = self.stub.PushTaskIns(request=req)
        return res

    def pull_task_res(self, req: PullTaskResRequest) -> PullTaskResResponse:
        """Get task results."""
        # Check if channel is open
        if self.stub is None:
            log(ERROR, ERROR_MESSAGE_DRIVER_NOT_CONNECTED)
            raise ConnectionError("`GrpcDriverStub` instance not connected")

        # Call Driver API
        res: PullTaskResResponse = self.stub.PullTaskRes(request=req)
        return res


class GrpcDriver(Driver):
    """`Driver` class provides an interface to the Driver API.

    Parameters
    ----------
    run_id : int
        The identifier of the run.
    stub : Optional[GrpcDriverStub] (default: None)
        The ``GrpcDriverStub`` instance used to communicate with the SuperLink.
        If None, an instance connected to "[::]:9091" will be created.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        run_id: int,
        stub: Optional[GrpcDriverStub] = None,
    ) -> None:
        self._run_id = run_id
        self._run: Optional[Run] = None
        self.stub = stub if stub is not None else GrpcDriverStub()
        self.node = Node(node_id=0, anonymous=True)

    @property
    def run(self) -> Run:
        """Run information."""
        self._get_stub_and_run_id()
        return Run(**vars(cast(Run, self._run)))

    def _get_stub_and_run_id(self) -> Tuple[GrpcDriverStub, int]:
        # Check if is initialized
        if self._run is None:
            # Connect
            if not self.stub.is_connected():
                self.stub.connect()
            # Get the run info
            req = GetRunRequest(run_id=self._run_id)
            res = self.stub.get_run(req)
            if not res.HasField("run"):
                raise RuntimeError(f"Cannot find the run with ID: {self._run_id}")
            self._run = Run(
                run_id=res.run.run_id,
                fab_id=res.run.fab_id,
                fab_version=res.run.fab_version,
                override_config=dict(res.run.override_config.items()),
            )

        return self.stub, self._run.run_id

    def _check_message(self, message: Message) -> None:
        # Check if the message is valid
        if not (
            message.metadata.run_id == cast(Run, self._run).run_id
            and message.metadata.src_node_id == self.node.node_id
            and message.metadata.message_id == ""
            and message.metadata.reply_to_message == ""
            and message.metadata.ttl > 0
        ):
            raise ValueError(f"Invalid message: {message}")

    def create_message(  # pylint: disable=too-many-arguments
        self,
        content: RecordSet,
        message_type: str,
        dst_node_id: int,
        group_id: str,
        ttl: Optional[float] = None,
    ) -> Message:
        """Create a new message with specified parameters.

        This method constructs a new `Message` with given content and metadata.
        The `run_id` and `src_node_id` will be set automatically.
        """
        _, run_id = self._get_stub_and_run_id()
        if ttl:
            warnings.warn(
                "A custom TTL was set, but note that the SuperLink does not enforce "
                "the TTL yet. The SuperLink will start enforcing the TTL in a future "
                "version of Flower.",
                stacklevel=2,
            )

        ttl_ = DEFAULT_TTL if ttl is None else ttl
        metadata = Metadata(
            run_id=run_id,
            message_id="",  # Will be set by the server
            src_node_id=self.node.node_id,
            dst_node_id=dst_node_id,
            reply_to_message="",
            group_id=group_id,
            ttl=ttl_,
            message_type=message_type,
        )
        return Message(metadata=metadata, content=content)

    def get_node_ids(self) -> List[int]:
        """Get node IDs."""
        stub, run_id = self._get_stub_and_run_id()
        # Call GrpcDriverStub method
        res = stub.get_nodes(GetNodesRequest(run_id=run_id))
        return [node.node_id for node in res.nodes]

    def push_messages(self, messages: Iterable[Message]) -> Iterable[str]:
        """Push messages to specified node IDs.

        This method takes an iterable of messages and sends each message
        to the node specified in `dst_node_id`.
        """
        stub, _ = self._get_stub_and_run_id()
        # Construct TaskIns
        task_ins_list: List[TaskIns] = []
        for msg in messages:
            # Check message
            self._check_message(msg)
            # Convert Message to TaskIns
            taskins = message_to_taskins(msg)
            # Add to list
            task_ins_list.append(taskins)
        # Call GrpcDriverStub method
        res = stub.push_task_ins(PushTaskInsRequest(task_ins_list=task_ins_list))
        return list(res.task_ids)

    def pull_messages(self, message_ids: Iterable[str]) -> Iterable[Message]:
        """Pull messages based on message IDs.

        This method is used to collect messages from the SuperLink that correspond to a
        set of given message IDs.
        """
        stub, _ = self._get_stub_and_run_id()
        # Pull TaskRes
        res = stub.pull_task_res(
            PullTaskResRequest(node=self.node, task_ids=message_ids)
        )
        # Convert TaskRes to Message
        msgs = [message_from_taskres(taskres) for taskres in res.task_res_list]
        return msgs

    def send_and_receive(
        self,
        messages: Iterable[Message],
        *,
        timeout: Optional[float] = None,
    ) -> Iterable[Message]:
        """Push messages to specified node IDs and pull the reply messages.

        This method sends a list of messages to their destination node IDs and then
        waits for the replies. It continues to pull replies until either all replies are
        received or the specified timeout duration is exceeded.
        """
        # Push messages
        msg_ids = set(self.push_messages(messages))

        # Pull messages
        end_time = time.time() + (timeout if timeout is not None else 0.0)
        ret: List[Message] = []
        while timeout is None or time.time() < end_time:
            res_msgs = self.pull_messages(msg_ids)
            ret.extend(res_msgs)
            msg_ids.difference_update(
                {msg.metadata.reply_to_message for msg in res_msgs}
            )
            if len(msg_ids) == 0:
                break
            # Sleep
            time.sleep(3)
        return ret

    def close(self) -> None:
        """Disconnect from the SuperLink if connected."""
        # Check if `connect` was called before
        if not self.stub.is_connected():
            return
        # Disconnect
        self.stub.disconnect()
