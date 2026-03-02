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

import grpc

from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.appio_token_auth_interceptor import (
    AppIoTokenAuthServerInterceptor,
    validate_method_requires_token_map,
)
from flwr.common.grpc import generic_create_grpc_server
from flwr.common.logger import log
from flwr.proto import serverappio_pb2  # pylint: disable=E0611
from flwr.proto.serverappio_pb2_grpc import (  # pylint: disable=E0611
    add_ServerAppIoServicer_to_server,
)
from flwr.server.superlink.linkstate import LinkStateFactory
from flwr.supercore.ffs import FfsFactory
from flwr.supercore.object_store import ObjectStoreFactory

from .serverappio_servicer import ServerAppIoServicer

SERVERAPPIO_METHOD_REQUIRES_TOKEN = {
    # Keep this table in sync with the proto service. Startup validation below
    # fails fast with a targeted error message if an RPC is added/renamed and no
    # explicit auth decision is recorded here.
    # SuperExec path (intentionally unauthenticated in this PR)
    "/flwr.proto.ServerAppIo/ListAppsToLaunch": False,
    "/flwr.proto.ServerAppIo/RequestToken": False,
    "/flwr.proto.ServerAppIo/GetRun": False,
    # App executor path (token required)
    "/flwr.proto.ServerAppIo/SendAppHeartbeat": True,
    "/flwr.proto.ServerAppIo/PullAppInputs": True,
    "/flwr.proto.ServerAppIo/PushAppOutputs": True,
    "/flwr.proto.ServerAppIo/GetNodes": True,
    "/flwr.proto.ServerAppIo/PushMessages": True,
    "/flwr.proto.ServerAppIo/PullMessages": True,
    "/flwr.proto.ServerAppIo/PushObject": True,
    "/flwr.proto.ServerAppIo/PullObject": True,
    "/flwr.proto.ServerAppIo/ConfirmMessageReceived": True,
    "/flwr.proto.ServerAppIo/UpdateRunStatus": True,
    "/flwr.proto.ServerAppIo/PushLogs": True,
}

validate_method_requires_token_map(
    service_name="ServerAppIo",
    package_name=serverappio_pb2.DESCRIPTOR.package,
    rpc_method_names=[
        method.name
        for method in serverappio_pb2.DESCRIPTOR.services_by_name["ServerAppIo"].methods
    ],
    method_requires_token=SERVERAPPIO_METHOD_REQUIRES_TOKEN,
    table_name="SERVERAPPIO_METHOD_REQUIRES_TOKEN",
    table_location="py/flwr/server/superlink/serverappio/serverappio_grpc.py",
)


def run_serverappio_api_grpc(
    address: str,
    state_factory: LinkStateFactory,
    ffs_factory: FfsFactory,
    objectstore_factory: ObjectStoreFactory,
    certificates: tuple[bytes, bytes, bytes] | None,
) -> grpc.Server:
    """Run ServerAppIo API (gRPC, request-response)."""
    # Create ServerAppIo API gRPC server
    serverappio_servicer: grpc.Server = ServerAppIoServicer(
        state_factory=state_factory,
        ffs_factory=ffs_factory,
        objectstore_factory=objectstore_factory,
    )
    serverappio_add_servicer_to_server_fn = add_ServerAppIoServicer_to_server
    interceptors = [
        AppIoTokenAuthServerInterceptor(
            state_provider=state_factory.state,
            method_requires_token=SERVERAPPIO_METHOD_REQUIRES_TOKEN,
        )
    ]
    serverappio_grpc_server = generic_create_grpc_server(
        servicer_and_add_fn=(
            serverappio_servicer,
            serverappio_add_servicer_to_server_fn,
        ),
        server_address=address,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        certificates=certificates,
        interceptors=interceptors,
    )

    log(INFO, "Flower Deployment Runtime: Starting ServerAppIo API on %s", address)
    serverappio_grpc_server.start()

    return serverappio_grpc_server
