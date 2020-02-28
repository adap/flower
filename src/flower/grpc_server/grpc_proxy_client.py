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
"""Networked Flower client implementation."""

from typing import Dict, Tuple

from flower import typing
from flower.client import Client
from flower.grpc_server.grpc_bridge import GRPCBridge
from flower.grpc_server.serde import ndarray_to_proto, proto_to_ndarray
from flower.proto.transport_pb2 import ServerMessage, Weights


class GRPCProxyClient(Client):
    """Client interface which delegates over the network."""

    def __init__(
        self, cid: str, info: Dict[str, str], bridge: GRPCBridge,
    ):
        super().__init__(cid, info)
        self.bridge = bridge

    def get_weights(self) -> typing.Weights:
        """Return the current local model weights"""
        return []

    def fit(self, weights: typing.Weights) -> Tuple[typing.Weights, int]:
        """Refine the provided weights using the locally held dataset."""

        weights_proto = [ndarray_to_proto(weight) for weight in weights]

        server_message = ServerMessage(
            fit=ServerMessage.Fit(weights=Weights(weights=weights_proto))
        )

        client_message = self.bridge.request(server_message)

        weights = [
            proto_to_ndarray(weight) for weight in client_message.fit.weights.weights
        ]
        num_examples = client_message.fit.num_examples

        return (weights, num_examples)

    def evaluate(self, weights: typing.Weights) -> Tuple[int, float]:
        """Evaluate the provided weights using the locally held dataset"""
        return (1, 1.0)
