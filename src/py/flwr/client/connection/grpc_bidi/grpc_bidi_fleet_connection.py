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
"""Connection for a gRPC bidirectional streaming channel to the SuperLink."""


from __future__ import annotations

import uuid
from collections.abc import Iterator
from logging import DEBUG, ERROR
from pathlib import Path
from queue import Queue
from types import TracebackType
from typing import cast

from cryptography.hazmat.primitives.asymmetric import ec

from flwr.common import (
    DEFAULT_TTL,
    GRPC_MAX_MESSAGE_LENGTH,
    ConfigsRecord,
    Message,
    Metadata,
    RecordSet,
)
from flwr.common import recordset_compat as compat
from flwr.common import serde
from flwr.common.constant import MessageType, MessageTypeLegacy
from flwr.common.grpc import create_channel, on_channel_state_change
from flwr.common.logger import log
from flwr.common.retry_invoker import RetryInvoker
from flwr.common.typing import Fab, Run
from flwr.proto.transport_pb2 import (  # pylint: disable=E0611
    ClientMessage,
    Reason,
    ServerMessage,
)
from flwr.proto.transport_pb2_grpc import FlowerServiceStub  # pylint: disable=E0611

from ..fleet_connection import FleetConnection

# The following flags can be uncommented for debugging. Other possible values:
# https://github.com/grpc/grpc/blob/master/doc/environment_variables.md
# import os
# os.environ["GRPC_VERBOSITY"] = "debug"
# os.environ["GRPC_TRACE"] = "tcp,http"


class GrpcBidiFleetConnection(FleetConnection):
    """Grpc-bidi fleet connection (will be deprecated)."""

    def __init__(  # pylint: disable=R0913, R0914, R0915, R0917
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
        """Initialize the GrpcBidiFleetConnection."""
        super().__init__(
            server_address=server_address,
            insecure=insecure,
            retry_invoker=retry_invoker,
            max_message_length=max_message_length,
            root_certificates=root_certificates,
            authentication_keys=authentication_keys,
        )

        if authentication_keys is not None:
            log(
                ERROR, "Client authentication is not supported for this transport type."
            )
        if isinstance(root_certificates, str):
            root_certificates = Path(root_certificates).read_bytes()

        channel = create_channel(
            server_address=server_address,
            insecure=insecure,
            root_certificates=root_certificates,
            max_message_length=max_message_length,
        )
        channel.subscribe(on_channel_state_change)

        queue: Queue[ClientMessage] = Queue(  # pylint: disable=unsubscriptable-object
            maxsize=1
        )
        stub = FlowerServiceStub(channel)

        server_message_iterator: Iterator[ServerMessage] = stub.Join(
            iter(queue.get, None)
        )

        self.channel = channel
        self.queue = queue
        self.server_message_iterator = server_message_iterator

    def ping(self) -> None:
        """Ping the SuperLink."""
        log(DEBUG, "Ping API is not supported by GrpcBidiConnection.")

    def create_node(self) -> int | None:
        """Request to create a node."""
        log(DEBUG, "CreateNode API is not supported by GrpcBidiConnection.")
        # gRPC-bidi doesn't have the concept of node_id,
        # so we return -1
        return -1

    def delete_node(self) -> None:
        """Request to delete a node."""
        log(DEBUG, "DeleteNode API is not supported by GrpcBidiConnection.")

    def receive(self) -> Message | None:
        """Receive a message."""  # Receive ServerMessage proto
        proto = next(self.server_message_iterator)

        # ServerMessage proto --> *Ins --> RecordSet
        field = proto.WhichOneof("msg")
        message_type = ""
        if field == "get_properties_ins":
            recordset = compat.getpropertiesins_to_recordset(
                serde.get_properties_ins_from_proto(proto.get_properties_ins)
            )
            message_type = MessageTypeLegacy.GET_PROPERTIES
        elif field == "get_parameters_ins":
            recordset = compat.getparametersins_to_recordset(
                serde.get_parameters_ins_from_proto(proto.get_parameters_ins)
            )
            message_type = MessageTypeLegacy.GET_PARAMETERS
        elif field == "fit_ins":
            recordset = compat.fitins_to_recordset(
                serde.fit_ins_from_proto(proto.fit_ins), False
            )
            message_type = MessageType.TRAIN
        elif field == "evaluate_ins":
            recordset = compat.evaluateins_to_recordset(
                serde.evaluate_ins_from_proto(proto.evaluate_ins), False
            )
            message_type = MessageType.EVALUATE
        elif field == "reconnect_ins":
            recordset = RecordSet()
            recordset.configs_records["config"] = ConfigsRecord(
                {"seconds": proto.reconnect_ins.seconds}
            )
            message_type = "reconnect"
        else:
            raise ValueError(
                "Unsupported instruction in ServerMessage, "
                "cannot deserialize from ProtoBuf"
            )

        # Construct Message
        return Message(
            metadata=Metadata(
                run_id=0,
                message_id=str(uuid.uuid4()),
                src_node_id=0,
                dst_node_id=0,
                reply_to_message="",
                group_id="",
                ttl=DEFAULT_TTL,
                message_type=message_type,
            ),
            content=recordset,
        )

    def send(self, message: Message) -> None:
        """Send a message."""
        # Retrieve RecordSet and message_type
        recordset = message.content
        message_type = message.metadata.message_type

        # RecordSet --> *Res --> *Res proto -> ClientMessage proto
        if message_type == MessageTypeLegacy.GET_PROPERTIES:
            getpropres = compat.recordset_to_getpropertiesres(recordset)
            msg_proto = ClientMessage(
                get_properties_res=serde.get_properties_res_to_proto(getpropres)
            )
        elif message_type == MessageTypeLegacy.GET_PARAMETERS:
            getparamres = compat.recordset_to_getparametersres(recordset, False)
            msg_proto = ClientMessage(
                get_parameters_res=serde.get_parameters_res_to_proto(getparamres)
            )
        elif message_type == MessageType.TRAIN:
            fitres = compat.recordset_to_fitres(recordset, False)
            msg_proto = ClientMessage(fit_res=serde.fit_res_to_proto(fitres))
        elif message_type == MessageType.EVALUATE:
            evalres = compat.recordset_to_evaluateres(recordset)
            msg_proto = ClientMessage(evaluate_res=serde.evaluate_res_to_proto(evalres))
        elif message_type == "reconnect":
            reason = cast(
                Reason.ValueType, recordset.configs_records["config"]["reason"]
            )
            msg_proto = ClientMessage(
                disconnect_res=ClientMessage.DisconnectRes(reason=reason)
            )
        else:
            raise ValueError(f"Invalid message type: {message_type}")

        # Send ClientMessage proto
        return self.queue.put(msg_proto, block=False)

    def get_run(self, run_id: int) -> Run:
        """Get run info."""
        log(DEBUG, "GetRun API is not supported by GrpcBidiConnection.")
        return Run.create_empty(run_id)

    def get_fab(self, fab_hash: str, run_id: int) -> Fab:
        """Get FAB file."""
        raise NotImplementedError

    def close(self) -> None:
        """Close the connection."""
        # Make sure to have a final
        self.channel.close()
        log(DEBUG, "gRPC channel closed")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType,
    ) -> None:
        """Exit from the context."""
        self.close()
