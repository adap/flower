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
"""SimulationIo gRPC API."""


from logging import INFO
from typing import Optional

import grpc

from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.logger import log
from flwr.proto.simulationio_pb2_grpc import (  # pylint: disable=E0611
    add_SimulationIoServicer_to_server,
)
from flwr.server.superlink.ffs.ffs_factory import FfsFactory
from flwr.server.superlink.linkstate import LinkStateFactory

from ..fleet.grpc_bidi.grpc_server import generic_create_grpc_server
from .simulationio_servicer import SimulationIoServicer


def run_simulationio_api_grpc(
    address: str,
    state_factory: LinkStateFactory,
    ffs_factory: FfsFactory,
    certificates: Optional[tuple[bytes, bytes, bytes]],
) -> grpc.Server:
    """Run SimulationIo API (gRPC, request-response)."""
    # Create SimulationIo API gRPC server
    simulationio_servicer: grpc.Server = SimulationIoServicer(
        state_factory=state_factory,
        ffs_factory=ffs_factory,
    )
    simulationio_add_servicer_to_server_fn = add_SimulationIoServicer_to_server
    simulationio_grpc_server = generic_create_grpc_server(
        servicer_and_add_fn=(
            simulationio_servicer,
            simulationio_add_servicer_to_server_fn,
        ),
        server_address=address,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        certificates=certificates,
    )

    log(
        INFO,
        "Flower Simulation Engine: Starting SimulationIo API on %s",
        address,
    )
    simulationio_grpc_server.start()

    return simulationio_grpc_server
