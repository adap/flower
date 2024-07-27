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
"""Driver gRPC API."""

from logging import INFO
from typing import Optional, Tuple

import grpc

from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.logger import log
from flwr.proto.driver_pb2_grpc import (  # pylint: disable=E0611
    add_DriverServicer_to_server,
)
from flwr.server.superlink.state import StateFactory

from ..fleet.grpc_bidi.grpc_server import generic_create_grpc_server
from .driver_servicer import DriverServicer


def run_driver_api_grpc(
    address: str,
    state_factory: StateFactory,
    certificates: Optional[Tuple[bytes, bytes, bytes]],
) -> grpc.Server:
    """Run Driver API (gRPC, request-response)."""
    # Create Driver API gRPC server
    driver_servicer: grpc.Server = DriverServicer(
        state_factory=state_factory,
    )
    driver_add_servicer_to_server_fn = add_DriverServicer_to_server
    driver_grpc_server = generic_create_grpc_server(
        servicer_and_add_fn=(driver_servicer, driver_add_servicer_to_server_fn),
        server_address=address,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        certificates=certificates,
    )

    log(INFO, "Flower ECE: Starting Driver API (gRPC-rere) on %s", address)
    driver_grpc_server.start()

    return driver_grpc_server
