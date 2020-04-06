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
"""This module contains functions for protobuf serialization and deserialization."""
# pylint: disable=missing-function-docstring
from io import BytesIO
from typing import Dict, Tuple, cast

import numpy as np

from flower import typing
from flower.proto.transport_pb2 import (
    ClientMessage,
    NDArray,
    Reason,
    ServerMessage,
    Weights,
)


def ndarray_to_proto(ndarray: np.ndarray) -> NDArray:
    """Serialize numpy array to NDArray protobuf message"""
    ndarray_bytes = BytesIO()
    np.save(ndarray_bytes, ndarray, allow_pickle=False)
    return NDArray(ndarray=ndarray_bytes.getvalue())


def proto_to_ndarray(ndarray_proto: NDArray) -> np.ndarray:
    """Deserialize NDArray protobuf message to a numpy array"""
    ndarray_bytes = BytesIO(ndarray_proto.ndarray)
    ndarray_deserialized = np.load(ndarray_bytes, allow_pickle=False)
    return cast(np.ndarray, ndarray_deserialized)


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


def server_get_weights_to_proto() -> ServerMessage.GetWeights:
    return ServerMessage.GetWeights()


# Not required:
# def server_get_weights_from_proto(msg: ServerMessage.GetWeights) -> None:


def client_get_weights_to_proto(weights: typing.Weights) -> ClientMessage.GetWeights:
    weights_proto = [ndarray_to_proto(weight) for weight in weights]
    return ClientMessage.GetWeights(weights=Weights(weights=weights_proto))


def client_get_weights_from_proto(msg: ClientMessage.GetWeights) -> typing.Weights:
    weights = [proto_to_ndarray(weight) for weight in msg.weights.weights]
    return weights


# === Fit messages ===


def fit_ins_to_proto(ins: typing.FitIns) -> ServerMessage.FitIns:
    """Serialize flower.FitIns to ProtoBuf message."""
    weights, config = ins
    weights_proto = [ndarray_to_proto(weight) for weight in weights]
    return ServerMessage.FitIns(weights=Weights(weights=weights_proto), config=config)


def fit_ins_from_proto(msg: ServerMessage.FitIns) -> typing.FitIns:
    """Deserialize flower.FitIns from ProtoBuf message."""
    weights = [proto_to_ndarray(weight) for weight in msg.weights.weights]
    config = msg.config
    return (weights, config)


def fit_res_to_proto(res: typing.FitRes) -> ClientMessage.FitRes:
    weights, num_examples = res
    weights_proto = [ndarray_to_proto(weight) for weight in weights]
    return ClientMessage.FitRes(
        weights=Weights(weights=weights_proto), num_examples=num_examples
    )


def fit_res_from_proto(msg: ClientMessage.FitRes) -> typing.FitRes:
    weights = [proto_to_ndarray(weight) for weight in msg.weights.weights]
    num_examples = msg.num_examples
    return weights, num_examples


# === Evaluate messages ===


def server_evaluate_to_proto(weights: typing.Weights) -> ServerMessage.Evaluate:
    weights_proto = [ndarray_to_proto(weight) for weight in weights]
    return ServerMessage.Evaluate(weights=Weights(weights=weights_proto))


def server_evaluate_from_proto(msg: ServerMessage.Evaluate) -> typing.Weights:
    weights = [proto_to_ndarray(weight) for weight in msg.weights.weights]
    return weights


def client_evaluate_to_proto(num_examples: int, loss: float) -> ClientMessage.Evaluate:
    return ClientMessage.Evaluate(num_examples=num_examples, loss=loss)


def client_evaluate_from_proto(msg: ClientMessage.Evaluate) -> Tuple[int, float]:
    return msg.num_examples, msg.loss


# === Property messages ===


def server_get_properties_to_proto() -> ServerMessage.GetProperties:
    return ServerMessage.GetProperties()


# Not required:
# def server_get_properties_from_proto(msg: ServerMessage.Evaluate) -> None:


def client_get_properties_to_proto(
    properties: Dict[str, str]
) -> ClientMessage.GetProperties:
    return ClientMessage.GetProperties(properties=properties)


def client_get_properties_from_proto(
    msg: ClientMessage.GetProperties,
) -> Dict[str, str]:
    return dict(msg.properties)
