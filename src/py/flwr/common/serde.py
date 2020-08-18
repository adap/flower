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
"""This module contains functions for protobuf serialization and
deserialization."""


from typing import List

from flwr.proto.transport_pb2 import ClientMessage, Parameters, Reason, ServerMessage

from . import typing

# pylint: disable=missing-function-docstring


def parameters_to_proto(parameters: typing.Parameters) -> Parameters:
    """."""
    return Parameters(tensors=parameters.tensors, tensor_type=parameters.tensor_type)


def parameters_from_proto(msg: Parameters) -> typing.Parameters:
    """."""
    tensors: List[bytes] = list(msg.tensors)
    return typing.Parameters(tensors=tensors, tensor_type=msg.tensor_type)


#  === Reconnect / Disconnect messages ===


def server_reconnect_to_proto(seconds: int) -> ServerMessage.Reconnect:
    return ServerMessage.Reconnect(seconds=seconds)


def server_reconnect_from_proto(msg: ServerMessage.Reconnect) -> int:
    return msg.seconds


def client_disconnect_to_proto(reason: str) -> ClientMessage.Disconnect:
    reason_proto = Reason.UNKNOWN
    if reason == "RECONNECT":
        reason_proto = Reason.RECONNECT
    elif reason == "POWER_DISCONNECTED":
        reason_proto = Reason.POWER_DISCONNECTED
    elif reason == "WIFI_UNAVAILABLE":
        reason_proto = Reason.WIFI_UNAVAILABLE

    return ClientMessage.Disconnect(reason=reason_proto)


def client_disconnect_from_proto(msg: ClientMessage.Disconnect) -> str:
    if msg.reason == Reason.RECONNECT:
        return "RECONNECT"
    if msg.reason == Reason.POWER_DISCONNECTED:
        return "POWER_DISCONNECTED"
    if msg.reason == Reason.WIFI_UNAVAILABLE:
        return "WIFI_UNAVAILABLE"
    return "UNKNOWN"


# === GetWeights messages ===


def get_parameters_to_proto() -> ServerMessage.GetParameters:
    """."""
    return ServerMessage.GetParameters()


# Not required:
# def get_weights_from_proto(msg: ServerMessage.GetWeights) -> None:


def parameters_res_to_proto(res: typing.ParametersRes) -> ClientMessage.ParametersRes:
    """."""
    parameters_proto = parameters_to_proto(res.parameters)
    return ClientMessage.ParametersRes(parameters=parameters_proto)


def parameters_res_from_proto(msg: ClientMessage.ParametersRes) -> typing.ParametersRes:
    """."""
    parameters = parameters_from_proto(msg.parameters)
    return typing.ParametersRes(parameters=parameters)


# === Fit messages ===


def fit_ins_to_proto(ins: typing.FitIns) -> ServerMessage.FitIns:
    """Serialize flower.FitIns to ProtoBuf message."""
    parameters, config = ins
    parameters_proto = parameters_to_proto(parameters)
    return ServerMessage.FitIns(parameters=parameters_proto, config=config)


def fit_ins_from_proto(msg: ServerMessage.FitIns) -> typing.FitIns:
    """Deserialize flower.FitIns from ProtoBuf message."""
    parameters = parameters_from_proto(msg.parameters)
    config = dict(msg.config)
    return (parameters, config)


def fit_res_to_proto(res: typing.FitRes) -> ClientMessage.FitRes:
    """Serialize flower.FitIns to ProtoBuf message."""
    parameters, num_examples, num_examples_ceil, fit_duration = res
    parameters_proto = parameters_to_proto(parameters)
    return ClientMessage.FitRes(
        parameters=parameters_proto,
        num_examples=num_examples,
        num_examples_ceil=num_examples_ceil,
        fit_duration=fit_duration,
    )


def fit_res_from_proto(msg: ClientMessage.FitRes) -> typing.FitRes:
    """Deserialize flower.FitRes from ProtoBuf message."""
    parameters = parameters_from_proto(msg.parameters)
    num_examples = msg.num_examples
    num_examples_ceil = msg.num_examples_ceil
    return parameters, num_examples, num_examples_ceil, msg.fit_duration


# === Evaluate messages ===


def evaluate_ins_to_proto(ins: typing.EvaluateIns) -> ServerMessage.EvaluateIns:
    """Serialize flower.EvaluateIns to ProtoBuf message."""
    parameters, config = ins
    parameters_proto = parameters_to_proto(parameters)
    return ServerMessage.EvaluateIns(parameters=parameters_proto, config=config)


def evaluate_ins_from_proto(msg: ServerMessage.EvaluateIns) -> typing.EvaluateIns:
    """Deserialize flower.EvaluateIns from ProtoBuf message."""
    parameters = parameters_from_proto(msg.parameters)
    config = dict(msg.config)
    return parameters, config


def evaluate_res_to_proto(res: typing.EvaluateRes) -> ClientMessage.EvaluateRes:
    """Serialize flower.EvaluateIns to ProtoBuf message."""
    num_examples, loss, acc = res
    return ClientMessage.EvaluateRes(num_examples=num_examples, loss=loss, accuracy=acc)


def evaluate_res_from_proto(msg: ClientMessage.EvaluateRes) -> typing.EvaluateRes:
    """Deserialize flower.EvaluateRes from ProtoBuf message."""
    return msg.num_examples, msg.loss, msg.accuracy
