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

from flwr.client.client import (
    Client,
    has_evaluate,
    has_fit,
    has_get_parameters,
    has_get_properties,
)
from flwr.common import serde, typing
from flwr.common.typing import Parameters
from flwr.proto.transport_pb2 import ClientMessage, Reason, ServerMessage

# pylint: disable=missing-function-docstring


class UnknownServerMessage(Exception):
    """Signifies that the received message is unknown."""


def handle(
    client: Client, server_msg: ServerMessage
) -> Tuple[ClientMessage, int, bool]:
    """Handle incoming messages from the server.

    Parameters
    ----------
    client : Client
        The Client instance provided by the user.

    Returns
    -------
    client_message: ClientMessage
        The message comming from the server, to be processed by the client.
    sleep_duration : int
        Number of seconds that the client should disconnect from the server.
    keep_going : bool
        Flag that indicates whether the client should continue to process the
        next message from the server (True) or disconnect and optionally
        reconnect later (False).
    """
    field = server_msg.WhichOneof("msg")
    if field == "reconnect_ins":
        disconnect_msg, sleep_duration = _reconnect(server_msg.reconnect_ins)
        return disconnect_msg, sleep_duration, False
    if field == "get_properties_ins":
        return _get_properties(client, server_msg.get_properties_ins), 0, True
    if field == "get_parameters_ins":
        return _get_parameters(client, server_msg.get_parameters_ins), 0, True
    if field == "fit_ins":
        return _fit(client, server_msg.fit_ins), 0, True
    if field == "evaluate_ins":
        return _evaluate(client, server_msg.evaluate_ins), 0, True
    raise UnknownServerMessage()


def _reconnect(
    reconnect_msg: ServerMessage.ReconnectIns,
) -> Tuple[ClientMessage, int]:
    # Determine the reason for sending DisconnectRes message
    reason = Reason.ACK
    sleep_duration = None
    if reconnect_msg.seconds is not None:
        reason = Reason.RECONNECT
        sleep_duration = reconnect_msg.seconds
    # Build DisconnectRes message
    disconnect_res = ClientMessage.DisconnectRes(reason=reason)
    return ClientMessage(disconnect_res=disconnect_res), sleep_duration


def _get_properties(
    client: Client, get_properties_msg: ServerMessage.GetPropertiesIns
) -> ClientMessage:
    # Check if client overrides get_properties
    if not has_get_properties(client=client):
        # If client does not override get_properties, don't call it
        get_properties_res = typing.GetPropertiesRes(
            status=typing.Status(
                code=typing.Code.GET_PROPERTIES_NOT_IMPLEMENTED,
                message="Client does not implement `get_properties`",
            ),
            properties={},
        )
        get_properties_res_proto = serde.get_properties_res_to_proto(get_properties_res)
        return ClientMessage(get_properties_res=get_properties_res_proto)

    # Deserialize get_properties instruction
    get_properties_ins = serde.get_properties_ins_from_proto(get_properties_msg)
    # Request properties
    get_properties_res = client.get_properties(get_properties_ins)
    # Serialize response
    get_properties_res_proto = serde.get_properties_res_to_proto(get_properties_res)
    return ClientMessage(get_properties_res=get_properties_res_proto)


def _get_parameters(
    client: Client, get_parameters_msg: ServerMessage.GetParametersIns
) -> ClientMessage:
    # Check if client overrides get_parameters
    if not has_get_parameters(client=client):
        # If client does not override get_parameters, don't call it
        get_parameters_res = typing.GetParametersRes(
            status=typing.Status(
                code=typing.Code.GET_PARAMETERS_NOT_IMPLEMENTED,
                message="Client does not implement `get_parameters`",
            ),
            parameters=Parameters(tensor_type="", tensors=[]),
        )
        get_parameters_res_proto = serde.get_parameters_res_to_proto(get_parameters_res)
        return ClientMessage(get_parameters_res=get_parameters_res_proto)

    # Deserialize get_properties instruction
    get_parameters_ins = serde.get_parameters_ins_from_proto(get_parameters_msg)
    # Request parameters
    get_parameters_res = client.get_parameters(get_parameters_ins)
    # Serialize response
    get_parameters_res_proto = serde.get_parameters_res_to_proto(get_parameters_res)
    return ClientMessage(get_parameters_res=get_parameters_res_proto)


def _fit(client: Client, fit_msg: ServerMessage.FitIns) -> ClientMessage:
    # Check if client overrides fit
    if not has_fit(client=client):
        # If client does not override fit, don't call it
        fit_res = typing.FitRes(
            status=typing.Status(
                code=typing.Code.FIT_NOT_IMPLEMENTED,
                message="Client does not implement `fit`",
            ),
            parameters=Parameters(tensor_type="", tensors=[]),
            num_examples=0,
            metrics={},
        )
        fit_res_proto = serde.fit_res_to_proto(fit_res)
        return ClientMessage(fit_res=fit_res_proto)

    # Deserialize fit instruction
    fit_ins = serde.fit_ins_from_proto(fit_msg)
    # Perform fit
    fit_res = client.fit(fit_ins)
    # Serialize fit result
    fit_res_proto = serde.fit_res_to_proto(fit_res)
    return ClientMessage(fit_res=fit_res_proto)


def _evaluate(client: Client, evaluate_msg: ServerMessage.EvaluateIns) -> ClientMessage:
    # Check if client overrides evaluate
    if not has_evaluate(client=client):
        # If client does not override evaluate, don't call it
        evaluate_res = typing.EvaluateRes(
            status=typing.Status(
                code=typing.Code.EVALUATE_NOT_IMPLEMENTED,
                message="Client does not implement `evaluate`",
            ),
            loss=0.0,
            num_examples=0,
            metrics={},
        )
        evaluate_res_proto = serde.evaluate_res_to_proto(evaluate_res)
        return ClientMessage(evaluate_res=evaluate_res_proto)

    # Deserialize evaluate instruction
    evaluate_ins = serde.evaluate_ins_from_proto(evaluate_msg)
    # Perform evaluation
    evaluate_res = client.evaluate(evaluate_ins)
    # Serialize evaluate result
    evaluate_res_proto = serde.evaluate_res_to_proto(evaluate_res)
    return ClientMessage(evaluate_res=evaluate_res_proto)
