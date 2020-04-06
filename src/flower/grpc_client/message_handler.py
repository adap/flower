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

# pylint: disable=missing-function-docstring


class UnkownServerMessage(Exception):
    """Signifies that the received message is unknown."""


def handle(client: Client, server_msg: ServerMessage) -> ClientMessage:
    if server_msg.HasField("reconnect"):
        raise UnkownServerMessage()
    if server_msg.HasField("get_weights"):
        return _get_weights(client)
    if server_msg.HasField("fit_ins"):
        return _fit(client, server_msg.fit_ins)
    if server_msg.HasField("evaluate_ins"):
        return _evaluate(client, server_msg.evaluate_ins)
    raise UnkownServerMessage()


def _get_weights(client: Client) -> ClientMessage:
    # No need to deserialize get_weights_msg as its empty
    weights = client.get_weights()
    weights_proto = serde.client_get_weights_to_proto(weights)
    return ClientMessage(get_weights=weights_proto)


def _fit(client: Client, fit_msg: ServerMessage.FitIns) -> ClientMessage:
    # Deserialize fit instruction
    fit_ins = serde.fit_ins_from_proto(fit_msg)
    # Perform fit
    fit_res = client.fit(fit_ins)
    # Serialize fit result
    fit_res_proto = serde.fit_res_to_proto(fit_res)
    return ClientMessage(fit_res=fit_res_proto)


def _evaluate(client: Client, evaluate_msg: ServerMessage.EvaluateIns) -> ClientMessage:
    # Deserialize evaluate instruction
    evaluate_ins = serde.evaluate_ins_from_proto(evaluate_msg)
    # Perform evaluation
    evaluate_res = client.evaluate(evaluate_ins)
    # Serialize evaluate result
    evaluate_res_proto = serde.evaluate_res_to_proto(evaluate_res)
    return ClientMessage(evaluate_res=evaluate_res_proto)
