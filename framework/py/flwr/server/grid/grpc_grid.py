# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Flower gRPC Grid."""


import time
from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from logging import DEBUG, ERROR, INFO, WARNING
from typing import Any, Optional, cast

import grpc
from tqdm import tqdm

from flwr.common import ArrayRecord, Message, RecordDict
from flwr.common.constant import (
    CHUNK_SIZE,
    SERVERAPPIO_API_DEFAULT_CLIENT_ADDRESS,
    SUPERLINK_NODE_ID,
)
from flwr.common.grpc import create_channel, on_channel_state_change
from flwr.common.logger import log, warn_deprecated_feature
from flwr.common.message import (
    allocate_byte_arrays,
    chunk_viewer,
    decouple_arrays_from_message,
    materialize_arrays,
    total_num_chunks,
)
from flwr.common.retry_invoker import _make_simple_grpc_retry_invoker, _wrap_stub
from flwr.common.serde import message_from_proto, message_to_proto, run_from_proto
from flwr.common.typing import Run
from flwr.proto.chunk_pb2 import (  # pylint: disable=E0611
    Chunk,
    PullChunkRequest,
    PullChunkResponse,
    PushChunkRequest,
    PushChunkResponse,
)
from flwr.proto.message_pb2 import Message as ProtoMessage  # pylint: disable=E0611
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse  # pylint: disable=E0611
from flwr.proto.serverappio_pb2 import (  # pylint: disable=E0611
    GetNodesRequest,
    GetNodesResponse,
    PullResMessagesRequest,
    PullResMessagesResponse,
    PushInsMessagesRequest,
    PushInsMessagesResponse,
)
from flwr.proto.serverappio_pb2_grpc import ServerAppIoStub  # pylint: disable=E0611

from .grid import Grid

ERROR_MESSAGE_PUSH_MESSAGES_RESOURCE_EXHAUSTED = """

[Grid.push_messages] gRPC error occurred:

The 2GB gRPC limit has been reached. Consider reducing the number of messages pushed
at once, or push messages individually, for example:

> msgs = [msg1, msg2, msg3]
> msg_ids = []
> for msg in msgs:
>     msg_id = grid.push_messages([msg])
>     msg_ids.extend(msg_id)
"""

ERROR_MESSAGE_PULL_MESSAGES_RESOURCE_EXHAUSTED = """

[Grid.pull_messages] gRPC error occurred:

The 2GB gRPC limit has been reached. Consider reducing the number of messages pulled
at once, or pull messages individually, for example:

> msgs_ids = [msg_id1, msg_id2, msg_id3]
> msgs = []
> for msg_id in msg_ids:
>     msg = grid.pull_messages([msg_id])
>     msgs.extend(msg)
"""


class GrpcGrid(Grid):
    """`GrpcGrid` provides an interface to the ServerAppIo API.

    Parameters
    ----------
    serverappio_service_address : str (default: "[::]:9091")
        The address (URL, IPv6, IPv4) of the SuperLink ServerAppIo API service.
    root_certificates : Optional[bytes] (default: None)
        The PEM-encoded root certificates as a byte string.
        If provided, a secure connection using the certificates will be
        established to an SSL-enabled Flower server.
    """

    _deprecation_warning_logged = False

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
        self.node = Node(node_id=SUPERLINK_NODE_ID)
        self._retry_invoker = _make_simple_grpc_retry_invoker()
        super().__init__()

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
        self._channel.subscribe(on_channel_state_change)
        self._grpc_stub = ServerAppIoStub(self._channel)
        _wrap_stub(self._grpc_stub, self._retry_invoker)
        log(DEBUG, "[flwr-serverapp] Connected to %s", self._addr)

    def _disconnect(self) -> None:
        """Disconnect from the ServerAppIo API."""
        if not self._is_connected:
            log(DEBUG, "Already disconnected")
            return
        channel: grpc.Channel = self._channel
        self._channel = None
        self._grpc_stub = None
        channel.close()
        log(DEBUG, "[flwr-serverapp] Disconnected")

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
            message.metadata.message_id == ""
            and message.metadata.reply_to_message_id == ""
            and message.metadata.ttl > 0
        ):
            raise ValueError(f"Invalid message: {message}")

    def create_message(  # pylint: disable=too-many-arguments,R0917
        self,
        content: RecordDict,
        message_type: str,
        dst_node_id: int,
        group_id: str,
        ttl: Optional[float] = None,
    ) -> Message:
        """Create a new message with specified parameters.

        This method constructs a new `Message` with given content and metadata.
        The `run_id` and `src_node_id` will be set automatically.
        """
        if not GrpcGrid._deprecation_warning_logged:
            GrpcGrid._deprecation_warning_logged = True
            warn_deprecated_feature(
                "`Driver.create_message` / `Grid.create_message` is deprecated."
                "Use `Message` constructor instead."
            )
        return Message(content, dst_node_id, message_type, ttl=ttl, group_id=group_id)

    def get_node_ids(self) -> Iterable[int]:
        """Get node IDs."""
        # Call GrpcServerAppIoStub method
        res: GetNodesResponse = self._stub.GetNodes(
            GetNodesRequest(run_id=cast(Run, self._run).run_id)
        )
        return [node.node_id for node in res.nodes]

    def push_messages(self, messages: Iterable[Message]) -> Iterable[str]:
        """Push messages to specified node IDs.

        This method takes an iterable of messages and sends each message
        to the node specified in `dst_node_id`.
        """
        # Construct Messages
        run_id = cast(Run, self._run).run_id
        message_proto_list: list[ProtoMessage] = []
        array_records: list[dict[str, ArrayRecord]] = []
        for msg in messages:
            # Populate metadata
            msg.metadata.__dict__["_run_id"] = run_id
            msg.metadata.__dict__["_src_node_id"] = self.node.node_id
            # Check message
            self._check_message(msg)

            # decouple array data in this message from the rest
            msg_, records = decouple_arrays_from_message(msg)

            # Store for later (will be send in chunks)
            array_records.append(records)

            # Convert to proto
            msg_proto = message_to_proto(msg_)
            # Add to list
            message_proto_list.append(msg_proto)

        try:
            # Call GrpcServerAppIoStub method
            res: PushInsMessagesResponse = self._stub.PushMessages(
                PushInsMessagesRequest(messages_list=message_proto_list, run_id=run_id)
            )
            if len([msg_id for msg_id in res.message_ids if msg_id]) != len(
                message_proto_list
            ):
                log(
                    WARNING,
                    "Not all messages could be pushed to the SuperLink. The returned "
                    "list has `None` for those messages (the order is preserved as "
                    "passed to `push_messages`). This could be due to a malformed "
                    "message.",
                )

            # Now push thde chunks
            for msg_id, msg_array_record in zip(res.message_ids, array_records):
                chunk_views = chunk_viewer(msg_array_record)

                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [
                        executor.submit(
                            self._push_chunk, chunk_view=chunk_view, message_id=msg_id
                        )
                        for chunk_view in chunk_views
                    ]

                    for future in tqdm(
                        as_completed(futures), total=len(futures), desc="PushChunk"
                    ):
                        _ = future.result()

            return list(res.message_ids)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:  # pylint: disable=E1101
                log(ERROR, ERROR_MESSAGE_PUSH_MESSAGES_RESOURCE_EXHAUSTED)
                return []
            raise

    def _push_chunk(
        self, message_id: str, chunk_view: list[dict[str, Any]]
    ) -> PushChunkResponse:

        # materialize Chunk
        chunk = Chunk(
            array_id=chunk_view["array_id"],
            record_id=chunk_view["record_id"],
            chunk_index=chunk_view["chunk_index"],
            data=chunk_view["data"].tobytes(),  # <----- materialize chunk (copies data)
        )
        # Push Chunk
        _: PushChunkResponse = self._stub.PushChunk(
            PushChunkRequest(chunks=[chunk], message_id=message_id, node=self.node)
        )

    def _pull_chunk(self, message_id: str) -> PullChunkResponse:
        """."""
        res: PullChunkResponse = self._stub.PullChunk(
            request=PullChunkRequest(message_id=message_id, node=self.node)
        )
        return res

    def _materialize_arays_in_message(self, message: Message) -> None:
        """."""
        # Allocate bytearrays
        bytearrays_dict = allocate_byte_arrays(msg_content=message.content)
        # Identify number of total chunks based on Array sizes
        total_chunks = total_num_chunks(msg_content=message.content)
        # Request one chunk and store in bytearray
        if total_chunks:
            # Request one chunk at a time
            log(INFO, f"Requesting {total_chunks} chunks!")

            inflight_futures = set()
            num_pulled_chunks = 0
            with (
                ThreadPoolExecutor(max_workers=4) as executor,
                tqdm(total=total_chunks, desc="PullChunk") as pbar,
            ):

                # Submit one request
                future = executor.submit(
                    self._pull_chunk, message_id=message.metadata.message_id
                )
                inflight_futures.add(future)

                while num_pulled_chunks < total_chunks:
                    done, inflight_futures = wait(
                        inflight_futures, return_when=FIRST_COMPLETED
                    )

                    for future in done:
                        response: PullChunkResponse = future.result()
                        if (
                            response.chunk.record_id
                        ):  # will be unset if a chunk wasn't available to pull
                            chunk: Chunk = response.chunk
                            # Place memory in the Array it belongs to
                            offset = CHUNK_SIZE * chunk.chunk_index
                            bytearrays_dict[chunk.record_id][chunk.array_id][
                                offset : offset + len(chunk.data)
                            ] = chunk.data
                            num_pulled_chunks += 1
                            pbar.update(1)  # we got a chunk, update progress bar

                        # Submit a new request if still needed
                        if num_pulled_chunks < total_chunks:
                            future_ = executor.submit(
                                self._pull_chunk, message_id=message.metadata.message_id
                            )
                            inflight_futures.add(future_)
                        else:
                            break

            # Put data in Message (i.e. materialize Message)
            materialize_arrays(
                msg_content=message.content, bytearray_dict=bytearrays_dict
            )

    def pull_messages(self, message_ids: Iterable[str]) -> Iterable[Message]:
        """Pull messages based on message IDs.

        This method is used to collect messages from the SuperLink that correspond to a
        set of given message IDs.
        """
        try:
            # Pull Messages
            res: PullResMessagesResponse = self._stub.PullMessages(
                PullResMessagesRequest(
                    message_ids=message_ids,
                    run_id=cast(Run, self._run).run_id,
                )
            )
            # Convert Message from Protobuf representation
            msgs = [message_from_proto(msg_proto) for msg_proto in res.messages_list]

            # With the above, we got the messages but the Arrays (if any) are empty
            # We need to pull the chunks
            for msg in msgs:
                self._materialize_arays_in_message(msg)

            return msgs
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:  # pylint: disable=E1101
                log(ERROR, ERROR_MESSAGE_PULL_MESSAGES_RESOURCE_EXHAUSTED)
                return []
            raise

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
                {msg.metadata.reply_to_message_id for msg in res_msgs}
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
