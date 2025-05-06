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
"""ServerAppIo gRPC API."""


from logging import INFO
from typing import Optional

import grpc

from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.grpc import generic_create_grpc_server
from flwr.common.logger import log
from flwr.proto.serverappio_pb2_grpc import (  # pylint: disable=E0611
    add_ServerAppIoServicer_to_server,
)
from flwr.server.superlink.ffs.ffs_factory import FfsFactory
from flwr.server.superlink.linkstate import LinkStateFactory

from .serverappio_servicer import ServerAppIoServicer


def run_serverappio_api_grpc(
    address: str,
    state_factory: LinkStateFactory,
    ffs_factory: FfsFactory,
    certificates: Optional[tuple[bytes, bytes, bytes]],
) -> grpc.Server:
    """Run ServerAppIo API (gRPC, request-response)."""
    # Create ServerAppIo API gRPC server
    serverappio_servicer: grpc.Server = ServerAppIoServicer(
        state_factory=state_factory,
        ffs_factory=ffs_factory,
    )
    serverappio_add_servicer_to_server_fn = add_ServerAppIoServicer_to_server
    serverappio_grpc_server = generic_create_grpc_server(
        servicer_and_add_fn=(
            serverappio_servicer,
            serverappio_add_servicer_to_server_fn,
        ),
        server_address=address,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        certificates=certificates,
    )

    log(INFO, "Flower ECE: Starting ServerAppIo API (gRPC-rere) on %s", address)
    serverappio_grpc_server.start()

    return serverappio_grpc_server
