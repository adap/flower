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

from typing import Tuple

from flower import typing
from flower.client import Client
from flower.grpc_server import serde
from flower.grpc_server.grpc_bridge import GRPCBridge
from flower.proto.transport_pb2 import ClientMessage, ServerMessage


class GRPCProxyClient(Client):
    """Client implementation which delegates over the network using gRPC."""

    def __init__(
        self, cid: str, bridge: GRPCBridge,
    ):
        super().__init__(cid)
        self.bridge = bridge

    def get_weights(self) -> typing.Weights:
        """Return the current local model weights"""
        get_weights_msg = serde.server_get_weights_to_proto()
        client_msg: ClientMessage = self.bridge.request(
            ServerMessage(get_weights=get_weights_msg)
        )
        weights = serde.client_get_weights_from_proto(client_msg.get_weights)
        return weights

    def fit(self, weights: typing.Weights) -> Tuple[typing.Weights, int]:
        """Refine the provided weights using the locally held dataset."""
        fit_msg = serde.server_fit_to_proto(weights)
        client_msg: ClientMessage = self.bridge.request(ServerMessage(fit=fit_msg))
        weights, num_examples = serde.client_fit_from_proto(client_msg.fit)
        return weights, num_examples

    def evaluate(self, weights: typing.Weights) -> Tuple[int, float]:
        """Evaluate the provided weights using the locally held dataset"""
        evaluate_msg = serde.server_evaluate_to_proto(weights)
        client_msg: ClientMessage = self.bridge.request(
            ServerMessage(evaluate=evaluate_msg)
        )
        num_examples, loss = serde.client_evaluate_from_proto(client_msg.evaluate)
        return num_examples, loss
