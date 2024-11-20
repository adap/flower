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
from typing import Optional

import grpc

from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.logger import log
from flwr.common.typing import UserConfig
from flwr.proto.exec_pb2_grpc import add_ExecServicer_to_server
from flwr.server.superlink.ffs.ffs_factory import FfsFactory
from flwr.server.superlink.fleet.grpc_bidi.grpc_server import generic_create_grpc_server
from flwr.server.superlink.linkstate import LinkStateFactory

from .exec_servicer import ExecServicer
from .executor import Executor


# pylint: disable-next=too-many-arguments, too-many-positional-arguments
def run_exec_api_grpc(
    address: str,
    executor: Executor,
    state_factory: LinkStateFactory,
    ffs_factory: FfsFactory,
    certificates: Optional[tuple[bytes, bytes, bytes]],
    config: UserConfig,
) -> grpc.Server:
    """Run Exec API (gRPC, request-response)."""
    executor.set_config(config)

    exec_servicer: grpc.Server = ExecServicer(
        linkstate_factory=state_factory,
        ffs_factory=ffs_factory,
        executor=executor,
    )
    exec_add_servicer_to_server_fn = add_ExecServicer_to_server
    exec_grpc_server = generic_create_grpc_server(
        servicer_and_add_fn=(exec_servicer, exec_add_servicer_to_server_fn),
        server_address=address,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        certificates=certificates,
    )

    log(INFO, "Flower Deployment Engine: Starting Exec API on %s", address)
    exec_grpc_server.start()

    return exec_grpc_server
