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
from collections.abc import Iterable
from logging import DEBUG, INFO, WARN, WARNING
from typing import Any, Optional, cast

import grpc

from flwr.common import DEFAULT_TTL, Message, Metadata, RecordSet
from flwr.common.constant import MAX_RETRY_DELAY, SERVERAPPIO_API_DEFAULT_CLIENT_ADDRESS
from flwr.common.grpc import create_channel
from flwr.common.logger import log
from flwr.common.retry_invoker import RetryInvoker, RetryState, exponential
from flwr.common.serde import message_from_taskres, message_to_taskins, run_from_proto
from flwr.common.typing import Run
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse  # pylint: disable=E0611
from flwr.proto.serverappio_pb2 import (  # pylint: disable=E0611
    GetNodesRequest,
    GetNodesResponse,
    PullTaskResRequest,
    PullTaskResResponse,
    PushTaskInsRequest,
    PushTaskInsResponse,
)
from flwr.proto.serverappio_pb2_grpc import ServerAppIoStub  # pylint: disable=E0611
from flwr.proto.task_pb2 import TaskIns  # pylint: disable=E0611

from .driver import Driver

ERROR_MESSAGE_DRIVER_NOT_CONNECTED = """
[Driver] Error: Not connected.

Call `connect()` on the `GrpcDriverStub` instance before calling any of the other
`GrpcDriverStub` methods.
"""


class GrpcDriver(Driver):
    """`GrpcDriver` provides an interface to the ServerAppIo API.

    Parameters
    ----------
    serverappio_service_address : str (default: "[::]:9091")
        The address (URL, IPv6, IPv4) of the SuperLink ServerAppIo API service.
    root_certificates : Optional[bytes] (default: None)
        The PEM-encoded root certificates as a byte string.
        If provided, a secure connection using the certificates will be
        established to an SSL-enabled Flower server.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        serverappio_service_address: str = SERVERAPPIO_API_DEFAULT_CLIENT_ADDRESS,
        root_certificates: Optional[bytes] = None,
    ) -> None:
        self._addr = serverappio_service_address
        self._cert = root_certificates
        self._run: Optional[Run] = None
        self._grpc_stub: Optional[ServerAppIoStub] = None
        self._channel: Optional[grpc.Channel] = None
        self.node = Node(node_id=0, anonymous=True)
        self._retry_invoker = _make_simple_grpc_retry_invoker()

    @property
    def _is_connected(self) -> bool:
        """Check if connected to the ServerAppIo API server."""
        return self._channel is not None

    def _connect(self) -> None:
        """Connect to the ServerAppIo API.

        This will not call GetRun.
        """
        if self._is_connected:
            log(WARNING, "Already connected")
            return
        self._channel = create_channel(
            server_address=self._addr,
            insecure=(self._cert is None),
            root_certificates=self._cert,
        )
        self._grpc_stub = ServerAppIoStub(self._channel)
        _wrap_stub(self._grpc_stub, self._retry_invoker)
        log(DEBUG, "[Driver] Connected to %s", self._addr)

    def _disconnect(self) -> None:
        """Disconnect from the ServerAppIo API."""
        if not self._is_connected:
            log(DEBUG, "Already disconnected")
            return
        channel: grpc.Channel = self._channel
        self._channel = None
        self._grpc_stub = None
        channel.close()
        log(DEBUG, "[Driver] Disconnected")

    def set_run(self, run_id: int) -> None:
        """Set the run."""
        # Get the run info
        req = GetRunRequest(run_id=run_id)
        res: GetRunResponse = self._stub.GetRun(req)
        if not res.HasField("run"):
            raise RuntimeError(f"Cannot find the run with ID: {run_id}")
        self._run = run_from_proto(res.run)

    @property
    def run(self) -> Run:
        """Run information."""
        return Run(**vars(self._run))

    @property
    def _stub(self) -> ServerAppIoStub:
        """ServerAppIo stub."""
        if not self._is_connected:
            self._connect()
        return cast(ServerAppIoStub, self._grpc_stub)

    def _check_message(self, message: Message) -> None:
        # Check if the message is valid
        if not (
            # Assume self._run being initialized
            message.metadata.run_id == cast(Run, self._run).run_id
            and message.metadata.src_node_id == self.node.node_id
            and message.metadata.message_id == ""
            and message.metadata.reply_to_message == ""
            and message.metadata.ttl > 0
        ):
            raise ValueError(f"Invalid message: {message}")

    def create_message(  # pylint: disable=too-many-arguments,R0917
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
        if ttl:
            warnings.warn(
                "A custom TTL was set, but note that the SuperLink does not enforce "
                "the TTL yet. The SuperLink will start enforcing the TTL in a future "
                "version of Flower.",
                stacklevel=2,
            )

        ttl_ = DEFAULT_TTL if ttl is None else ttl
        metadata = Metadata(
            run_id=cast(Run, self._run).run_id,
            message_id="",  # Will be set by the server
            src_node_id=self.node.node_id,
            dst_node_id=dst_node_id,
            reply_to_message="",
            group_id=group_id,
            ttl=ttl_,
            message_type=message_type,
        )
        return Message(metadata=metadata, content=content)

    def get_node_ids(self) -> list[int]:
        """Get node IDs."""
        # Call GrpcDriverStub method
        res: GetNodesResponse = self._stub.GetNodes(
            GetNodesRequest(run_id=cast(Run, self._run).run_id)
        )
        return [node.node_id for node in res.nodes]

    def push_messages(self, messages: Iterable[Message]) -> Iterable[str]:
        """Push messages to specified node IDs.

        This method takes an iterable of messages and sends each message
        to the node specified in `dst_node_id`.
        """
        # Construct TaskIns
        task_ins_list: list[TaskIns] = []
        for msg in messages:
            # Check message
            self._check_message(msg)
            # Convert Message to TaskIns
            taskins = message_to_taskins(msg)
            # Add to list
            task_ins_list.append(taskins)
        # Call GrpcDriverStub method
        res: PushTaskInsResponse = self._stub.PushTaskIns(
            PushTaskInsRequest(
                task_ins_list=task_ins_list, run_id=cast(Run, self._run).run_id
            )
        )
        return list(res.task_ids)

    def pull_messages(self, message_ids: Iterable[str]) -> Iterable[Message]:
        """Pull messages based on message IDs.

        This method is used to collect messages from the SuperLink that correspond to a
        set of given message IDs.
        """
        # Pull TaskRes
        res: PullTaskResResponse = self._stub.PullTaskRes(
            PullTaskResRequest(
                node=self.node, task_ids=message_ids, run_id=cast(Run, self._run).run_id
            )
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
        ret: list[Message] = []
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
        if not self._is_connected:
            return
        # Disconnect
        self._disconnect()


def _make_simple_grpc_retry_invoker() -> RetryInvoker:
    """Create a simple gRPC retry invoker."""

    def _on_sucess(retry_state: RetryState) -> None:
        if retry_state.tries > 1:
            log(
                INFO,
                "Connection successful after %.2f seconds and %s tries.",
                retry_state.elapsed_time,
                retry_state.tries,
            )

    def _on_backoff(retry_state: RetryState) -> None:
        if retry_state.tries == 1:
            log(WARN, "Connection attempt failed, retrying...")
        else:
            log(
                WARN,
                "Connection attempt failed, retrying in %.2f seconds",
                retry_state.actual_wait,
            )

    def _on_giveup(retry_state: RetryState) -> None:
        if retry_state.tries > 1:
            log(
                WARN,
                "Giving up reconnection after %.2f seconds and %s tries.",
                retry_state.elapsed_time,
                retry_state.tries,
            )

    return RetryInvoker(
        wait_gen_factory=lambda: exponential(max_delay=MAX_RETRY_DELAY),
        recoverable_exceptions=grpc.RpcError,
        max_tries=None,
        max_time=None,
        on_success=_on_sucess,
        on_backoff=_on_backoff,
        on_giveup=_on_giveup,
        should_giveup=lambda e: e.code() != grpc.StatusCode.UNAVAILABLE,  # type: ignore
    )


def _wrap_stub(stub: ServerAppIoStub, retry_invoker: RetryInvoker) -> None:
    """Wrap the gRPC stub with a retry invoker."""

    def make_lambda(original_method: Any) -> Any:
        return lambda *args, **kwargs: retry_invoker.invoke(
            original_method, *args, **kwargs
        )

    for method_name in vars(stub):
        method = getattr(stub, method_name)
        if callable(method):
            setattr(stub, method_name, make_lambda(method))
