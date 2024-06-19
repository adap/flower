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
"""SuperExec gRPC API."""

from logging import INFO
from typing import Optional, Tuple

import grpc

from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.logger import log
from flwr.proto.exec_pb2_grpc import add_ExecServicer_to_server
from flwr.server.superlink.fleet.grpc_bidi.grpc_server import generic_create_grpc_server

from .exec_servicer import ExecServicer
from .executor import Executor


def run_superexec_api_grpc(
    address: str,
    executor: Executor,
    certificates: Optional[Tuple[bytes, bytes, bytes]],
) -> grpc.Server:
    """Run SuperExec API (gRPC, request-response)."""
    exec_servicer: grpc.Server = ExecServicer(
        executor=executor,
    )
    superexec_add_servicer_to_server_fn = add_ExecServicer_to_server
    superexec_grpc_server = generic_create_grpc_server(
        servicer_and_add_fn=(exec_servicer, superexec_add_servicer_to_server_fn),
        server_address=address,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        certificates=certificates,
    )

    log(INFO, "Flower ECE: Starting SuperExec API (gRPC-rere) on %s", address)
    superexec_grpc_server.start()

    return superexec_grpc_server