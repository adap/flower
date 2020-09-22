# Copyright 2020 Adap GmbH. All Rights Reserved.
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
"""Implements utility function to create a grpc server."""
import concurrent.futures

import grpc

from flwr.proto import transport_pb2_grpc
from flwr.server.client_manager import ClientManager
from flwr.server.grpc_server import flower_service_servicer as fss


def start_insecure_grpc_server(
    client_manager: ClientManager,
    server_address: str,
    max_concurrent_workers: int = 1000,
) -> grpc.Server:
    """Create grpc server and return registered FlowerServiceServicer instance.

    If used in a main function server.wait_for_termination(timeout=None)
    should be called as otherwise the server will immediately stop.
    """
    server = grpc.server(
        concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_workers),
        maximum_concurrent_rpcs=max_concurrent_workers,
        options=[
            ("grpc.max_concurrent_streams", max(100, max_concurrent_workers)),
            ("grpc.max_send_message_length", 256 * 1024 * 1024),
            ("grpc.max_receive_message_length", 256 * 1024 * 1024),
        ],
    )

    servicer = fss.FlowerServiceServicer(client_manager)
    transport_pb2_grpc.add_FlowerServiceServicer_to_server(  # type: ignore
        servicer, server
    )

    server.add_insecure_port(server_address)
    server.start()

    return server
