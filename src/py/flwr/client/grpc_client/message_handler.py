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


from typing import Tuple

from flwr.client.client import Client
from flwr.common import serde
from flwr.proto.transport_pb2 import ClientMessage, Reason, ServerMessage

# pylint: disable=missing-function-docstring


class UnknownServerMessage(Exception):
    """Signifies that the received message is unknown."""


def handle(
    client: Client, server_msg: ServerMessage
) -> Tuple[ClientMessage, int, bool]:
    if server_msg.HasField("reconnect"):
        disconnect_msg, sleep_duration = _reconnect(server_msg.reconnect)
        return disconnect_msg, sleep_duration, False
    if server_msg.HasField("get_parameters"):
        return _get_parameters(client), 0, True
    if server_msg.HasField("fit_ins"):
        return _fit(client, server_msg.fit_ins), 0, True
    if server_msg.HasField("evaluate_ins"):
        return _evaluate(client, server_msg.evaluate_ins), 0, True
    if server_msg.HasField("properties_ins"):
        return _get_properties(client, server_msg.properties_ins), 0, True
    raise UnknownServerMessage()


def _get_parameters(client: Client) -> ClientMessage:
    # No need to deserialize get_parameters_msg (it's empty)
    parameters_res = client.get_parameters()
    parameters_res_proto = serde.parameters_res_to_proto(parameters_res)
    return ClientMessage(parameters_res=parameters_res_proto)


def _get_properties(
    client: Client, properties_msg: ServerMessage.PropertiesIns
) -> ClientMessage:
    # Deserialize get_properties instruction
    properties_ins = serde.properties_ins_from_proto(properties_msg)
    # Request for properties
    properties_res = client.get_properties(properties_ins)
    # Serialize response
    properties_res_proto = serde.properties_res_to_proto(properties_res)
    return ClientMessage(properties_res=properties_res_proto)


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


def _reconnect(
    reconnect_msg: ServerMessage.Reconnect,
) -> Tuple[ClientMessage, int]:
    # Determine the reason for sending Disconnect message
    reason = Reason.ACK
    sleep_duration = None
    if reconnect_msg.seconds is not None:
        reason = Reason.RECONNECT
        sleep_duration = reconnect_msg.seconds
    # Build Disconnect message
    disconnect = ClientMessage.Disconnect(reason=reason)
    return ClientMessage(disconnect=disconnect), sleep_duration
