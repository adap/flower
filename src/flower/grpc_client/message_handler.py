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
"""Handle server messages by calling appropriate client methods."""
from flower.client import Client
from flower.grpc_server import serde
from flower.proto.transport_pb2 import ClientMessage, ServerMessage
from logging import DEBUG
from flower.logger import log

# pylint: disable=missing-function-docstring


class UnkownServerMessage(Exception):
    """Signifies that the received message is unknown."""


def handle(client: Client, server_msg: ServerMessage) -> ClientMessage:
    if server_msg.HasField("reconnect"):
        log(DEBUG, "ReconnetMessage")
        raise UnkownServerMessage()
    if server_msg.HasField("get_weights"):
        log(DEBUG, "GetWeightsMessage")
        return _get_weights(client)
    if server_msg.HasField("fit"):
        log(DEBUG, "FitMessage")
        return _fit(client, server_msg.fit)
    if server_msg.HasField("evaluate"):
        log(DEBUG, "EvaluateMessage")
        return _evaluate(client, server_msg.evaluate)

    log(DEBUG, "raising UnkownServerMessage()")
    raise UnkownServerMessage()


def _get_weights(client: Client) -> ClientMessage:
    # No need to deserialize get_weights_msg as its empty
    log(DEBUG, "client.get_weights()")
    weights = client.get_weights()
    log(DEBUG, "serde.client_get_weights_to_proto")
    weights_proto = serde.client_get_weights_to_proto(weights)
    log(DEBUG, "ClientMessage(get_weights=weights_proto)")
    return ClientMessage(get_weights=weights_proto)


def _fit(client: Client, fit_msg: ServerMessage.Fit) -> ClientMessage:
    log(DEBUG, "serde.server_fit_from_proto")
    weights = serde.server_fit_from_proto(fit_msg)
    log(DEBUG, "client.fit")
    weights_prime, num_examples = client.fit(weights)
    log(DEBUG, "serde.client_fit_to_proto")
    fit_proto = serde.client_fit_to_proto(weights_prime, num_examples)
    log(DEBUG, "ClientMessage(fit=fit_proto)")
    return ClientMessage(fit=fit_proto)


def _evaluate(client: Client, evaluate_msg: ServerMessage.Evaluate) -> ClientMessage:
    log(DEBUG, "serde.server_evaluate_from_proto")
    weights = serde.server_evaluate_from_proto(evaluate_msg)
    log(DEBUG, "client.evaluate")
    num_examples, loss = client.evaluate(weights)
    log(DEBUG, "serde.client_evaluate_to_proto")
    evaluate_proto = serde.client_evaluate_to_proto(num_examples, loss)
    log(DEBUG, "return ClientMessage(evaluate=evaluate_proto)")
    return ClientMessage(evaluate=evaluate_proto)
