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
"""Servicer for FlowerService."""
from typing import Iterator

import grpc

from flower.client_manager import ClientManager
from flower.proto import transport_pb2_grpc
from flower.proto.transport_pb2 import ClientRequest, ServerResponse


class FlowerServiceServicer(transport_pb2_grpc.FlowerServiceServicer):
    """FlowerServiceServicer for bi-directional gRPC instruction stream."""

    def __init__(self, client_manager: ClientManager) -> None:
        self.client_manager = client_manager

    def Join(  # pylint: disable=invalid-name
        self, request_iterator: Iterator[ClientRequest], context: grpc.ServicerContext
    ) -> Iterator[ServerResponse]:
        for _ in request_iterator:
            # Yield empty message
            yield ServerResponse()
