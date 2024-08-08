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
"""Flower background ClientApp."""

from logging import DEBUG, ERROR, INFO

import grpc

# from flwr.cli.install import install_from_fab
from flwr.client.client_app import ClientApp
from flwr.common.grpc import GRPC_MAX_MESSAGE_LENGTH, create_channel
from flwr.common.logger import log
from flwr.common.serde import (
    context_from_proto,
    context_to_proto,
    message_from_proto,
    message_to_proto,
    run_from_proto,
)

# pylint: disable=E0401,E0611
from flwr.proto.appio_pb2 import PullClientAppInputsRequest, PushClientAppOutputsRequest
from flwr.proto.appio_pb2_grpc import ClientAppIoStub, add_ClientAppIoServicer_to_server

# pylint: disable=E0611
from flwr.server.superlink.fleet.grpc_bidi.grpc_server import generic_create_grpc_server

from .clientappio_servicer import ClientAppIoServicer
from .utils import _get_load_client_app_fn


def _run_background_client(  # pylint: disable=R0914
    address: str,
    token: int,
) -> None:
    """Run background Flower ClientApp process."""

    def on_channel_state_change(channel_connectivity: str) -> None:
        """Log channel connectivity."""
        log(DEBUG, channel_connectivity)

    channel = create_channel(
        server_address=address,
        insecure=True,
    )
    channel.subscribe(on_channel_state_change)

    try:
        stub = ClientAppIoStub(channel)

        req = PullClientAppInputsRequest(token=token)
        res = stub.PullClientAppInputs(req)
        # fab_file = res.fab
        run = run_from_proto(res.run)
        message = message_from_proto(res.message)
        context = context_from_proto(res.context)
        # Ensures FAB is installed (default is Flower directory)
        # install_from_fab(
        #     fab_file, None, True
        # )
        load_client_app_fn = _get_load_client_app_fn(
            default_app_ref="",
            project_dir="",
            multi_app=True,
            flwr_dir=None,
        )
        # print(f"FAB ID: {run.fab_id}, FAB version: {run.fab_version}")
        client_app: ClientApp = load_client_app_fn(
            run.fab_id, run.fab_version  # To be optimized later
        )
        # Execute ClientApp
        reply_message = client_app(message=message, context=context)

        proto_message = message_to_proto(reply_message)
        proto_context = context_to_proto(context)
        req = PushClientAppOutputsRequest(
            token=token,
            message=proto_message,
            context=proto_context,
        )
        res = stub.PushClientAppOutputs(req)
    except KeyboardInterrupt:
        log(INFO, "Closing connection")
    except grpc.RpcError as e:
        log(ERROR, "GRPC error occurred: %s", str(e))
    finally:
        channel.close()


def run_clientappio_api_grpc(
    address: str = "0.0.0.0:9094",
) -> tuple[grpc.Server, grpc.Server]:
    """Run ClientAppIo API (gRPC-rere)."""
    clientappio_servicer: grpc.Server = ClientAppIoServicer()
    clientappio_add_servicer_to_server_fn = add_ClientAppIoServicer_to_server
    clientappio_grpc_server = generic_create_grpc_server(
        servicer_and_add_fn=(
            clientappio_servicer,
            clientappio_add_servicer_to_server_fn,
        ),
        server_address=address,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
    )
    log(INFO, "Starting Flower ClientAppIo gRPC server on %s", address)
    clientappio_grpc_server.start()
    return clientappio_servicer, clientappio_grpc_server
