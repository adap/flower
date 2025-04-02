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
from flwr.common.auth_plugin import ExecAuthPlugin
from flwr.common.event_log_plugin import EventLogWriterPlugin
from flwr.common.grpc import generic_create_grpc_server
from flwr.common.logger import log
from flwr.common.typing import UserConfig
from flwr.proto.exec_pb2_grpc import add_ExecServicer_to_server
from flwr.server.superlink.ffs.ffs_factory import FfsFactory
from flwr.server.superlink.linkstate import LinkStateFactory
from flwr.superexec.exec_event_log_interceptor import ExecEventLogInterceptor
from flwr.superexec.exec_user_auth_interceptor import ExecUserAuthInterceptor

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
    auth_plugin: Optional[ExecAuthPlugin] = None,
    event_log_plugin: Optional[EventLogWriterPlugin] = None,
) -> grpc.Server:
    """Run Exec API (gRPC, request-response)."""
    executor.set_config(config)

    exec_servicer: grpc.Server = ExecServicer(
        linkstate_factory=state_factory,
        ffs_factory=ffs_factory,
        executor=executor,
        auth_plugin=auth_plugin,
    )
    interceptors: list[grpc.ServerInterceptor] = []
    if auth_plugin is not None:
        interceptors.append(ExecUserAuthInterceptor(auth_plugin))
    # Event log interceptor must be added after user auth interceptor
    if event_log_plugin is not None:
        interceptors.append(ExecEventLogInterceptor(event_log_plugin))
        log(INFO, "Flower event logging enabled")
    exec_add_servicer_to_server_fn = add_ExecServicer_to_server
    exec_grpc_server = generic_create_grpc_server(
        servicer_and_add_fn=(exec_servicer, exec_add_servicer_to_server_fn),
        server_address=address,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        certificates=certificates,
        interceptors=interceptors or None,
    )

    if auth_plugin is None:
        log(INFO, "Flower Deployment Engine: Starting Exec API on %s", address)
    else:
        log(
            INFO,
            "Flower Deployment Engine: Starting Exec API with user "
            "authentication on %s",
            address,
        )
    exec_grpc_server.start()

    return exec_grpc_server
