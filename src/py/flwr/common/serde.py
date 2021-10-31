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
"""ProtoBuf serialization and deserialization."""


from typing import Any, List, cast

from flwr.proto.transport_pb2 import (
    ClientMessage,
    Parameters,
    Reason,
    Scalar,
    ServerMessage,
)
from numpy.core import records

from . import typing, parameter
import pickle
import os
import numpy as np

# pylint: disable=missing-function-docstring


def parameters_to_proto(parameters: typing.Parameters) -> Parameters:
    """."""
    return Parameters(tensors=parameters.tensors, tensor_type=parameters.tensor_type)


def parameters_from_proto(msg: Parameters) -> typing.Parameters:
    """."""
    tensors: List[bytes] = list(msg.tensors)
    return typing.Parameters(tensors=tensors, tensor_type=msg.tensor_type)


#  === Reconnect message ===


def reconnect_to_proto(reconnect: typing.Reconnect) -> ServerMessage.Reconnect:
    """Serialize flower.Reconnect to ProtoBuf message."""
    if reconnect.seconds is not None:
        return ServerMessage.Reconnect(seconds=reconnect.seconds)
    return ServerMessage.Reconnect()


def reconnect_from_proto(msg: ServerMessage.Reconnect) -> typing.Reconnect:
    """Deserialize flower.Reconnect from ProtoBuf message."""
    return typing.Reconnect(seconds=msg.seconds)


# === Disconnect message ===


def disconnect_to_proto(disconnect: typing.Disconnect) -> ClientMessage.Disconnect:
    """Serialize flower.Disconnect to ProtoBuf message."""
    reason_proto = Reason.UNKNOWN
    if disconnect.reason == "RECONNECT":
        reason_proto = Reason.RECONNECT
    elif disconnect.reason == "POWER_DISCONNECTED":
        reason_proto = Reason.POWER_DISCONNECTED
    elif disconnect.reason == "WIFI_UNAVAILABLE":
        reason_proto = Reason.WIFI_UNAVAILABLE
    return ClientMessage.Disconnect(reason=reason_proto)


def disconnect_from_proto(msg: ClientMessage.Disconnect) -> typing.Disconnect:
    """Deserialize flower.Disconnect from ProtoBuf message."""
    if msg.reason == Reason.RECONNECT:
        return typing.Disconnect(reason="RECONNECT")
    if msg.reason == Reason.POWER_DISCONNECTED:
        return typing.Disconnect(reason="POWER_DISCONNECTED")
    if msg.reason == Reason.WIFI_UNAVAILABLE:
        return typing.Disconnect(reason="WIFI_UNAVAILABLE")
    return typing.Disconnect(reason="UNKNOWN")


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
    ###
    # Clean the directory
    for f in os.listdir("data"):
        os.remove(os.path.join("data", f))
    # Create shape file
    weights = parameter.parameters_to_weights(parameters)
    shape_file = open(os.path.join("data", "shapes.txt"), "w")
    for weight in weights:
        shape = tuple(weight.shape)
        shape_str = ""
        for num in shape:
            shape_str += str(num) + ","
        shape_str = shape_str[:len(shape_str) - 1] + "\n"
        shape_file.write(shape_str)
    shape_file.close()
    ###
    return typing.ParametersRes(parameters=parameters)


# === Fit messages ===


def fit_ins_to_proto(ins: typing.FitIns, cid: str) -> ServerMessage.FitIns:
    """Serialize flower.FitIns to ProtoBuf message."""
    ###
    # Dump data
    # Create file name
    i = 1
    file_name = "client" + cid[11:] + "fit_ins" + str(i) + ".f64"
    files = [f for f in os.listdir("data")]
    while file_name in files:
        i += 1
        file_name = "client" + cid[11:] + "fit_ins" + str(i) + ".f64"
    # Decode bytes into array
    weights = parameter.parameters_to_weights(ins.parameters)
    # Flatten parameters into 1D array of 64-bit floats
    out_array = weights[0]
    out_array = out_array.flatten("C")
    for j in range(1, len(weights)):
        nparray = weights[j]
        nparray = nparray.flatten("C")
        out_array = np.concatenate((out_array, nparray))
    out_array = out_array.astype("float64")
    # Save array to .f64 file
    out_array.tofile(os.path.join("data", file_name))
    
    
    ###
    parameters_proto = parameters_to_proto(ins.parameters)
    config_msg = metrics_to_proto(ins.config)
    return ServerMessage.FitIns(parameters=parameters_proto, config=config_msg)


def fit_ins_from_proto(msg: ServerMessage.FitIns, name: str) -> typing.FitIns:
    """Deserialize flower.FitIns from ProtoBuf message."""
    parameters = parameters_from_proto(msg.parameters)
    config = metrics_from_proto(msg.config)
    return typing.FitIns(parameters=parameters, config=config)


def fit_res_to_proto(res: typing.FitRes, name: str) -> ClientMessage.FitRes:
    """Serialize flower.FitIns to ProtoBuf message."""
    ###
    i = 1
    file_name = "client" + name + "fit_res" + str(i) + ".f32"
    files = [f for f in os.listdir("data")]
    while file_name in files:
        i += 1
        file_name = "client" + name + "fit_res" + str(i) + ".f32"
    # Decode bytes into array
    weights = parameter.parameters_to_weights(res.parameters)
    # Flatten parameters into 1D array of 32-bit floats
    out_array = weights[0]
    out_array = out_array.flatten("C")
    for i in range(1, len(weights)):
        nparray = weights[i]
        nparray = nparray.flatten("C")
        out_array = np.concatenate((out_array, nparray))
    # Save array to .f32 file
    out_array.tofile(os.path.join("data", file_name))
    ###
    parameters_proto = parameters_to_proto(res.parameters)
    metrics_msg = None if res.metrics is None else metrics_to_proto(res.metrics)
    # Legacy case, will be removed in a future release
    if res.num_examples_ceil is not None and res.fit_duration is not None:
        return ClientMessage.FitRes(
            parameters=parameters_proto,
            num_examples=res.num_examples,
            num_examples_ceil=res.num_examples_ceil,  # Deprecated
            fit_duration=res.fit_duration,  # Deprecated
            metrics=metrics_msg,
        )
    # Legacy case, will be removed in a future release
    if res.num_examples_ceil is not None:
        return ClientMessage.FitRes(
            parameters=parameters_proto,
            num_examples=res.num_examples,
            num_examples_ceil=res.num_examples_ceil,  # Deprecated
            metrics=metrics_msg,
        )
    # Legacy case, will be removed in a future release
    if res.fit_duration is not None:
        return ClientMessage.FitRes(
            parameters=parameters_proto,
            num_examples=res.num_examples,
            fit_duration=res.fit_duration,  # Deprecated
            metrics=metrics_msg,
        )
    # Forward-compatible case
    return ClientMessage.FitRes(
        parameters=parameters_proto,
        num_examples=res.num_examples,
        metrics=metrics_msg,
    )


def fit_res_from_proto(msg: ClientMessage.FitRes, cid: str) -> typing.FitRes:
    """Deserialize flower.FitRes from ProtoBuf message."""
    parameters = parameters_from_proto(msg.parameters)
    metrics = None if msg.metrics is None else metrics_from_proto(msg.metrics)
    return typing.FitRes(
        parameters=parameters,
        num_examples=msg.num_examples,
        num_examples_ceil=msg.num_examples_ceil,  # Deprecated
        fit_duration=msg.fit_duration,  # Deprecated
        metrics=metrics,
    )


# === Evaluate messages ===


def evaluate_ins_to_proto(ins: typing.EvaluateIns, cid: str) -> ServerMessage.EvaluateIns:
    """Serialize flower.EvaluateIns to ProtoBuf message."""
    ###
    weights = parameter.parameters_to_weights(ins.parameters)
    i = 1
    file_name = "client" + cid[11:] + "eval_ins" + str(i) + ".f64"
    files = [f for f in os.listdir("data")]
    while file_name in files:
        i += 1
        file_name = "client" + cid[11:] + "eval_ins" + str(i) + ".f64"
    # Decode bytes into array
    weights = parameter.parameters_to_weights(ins.parameters)
    # Flatten parameters into 1D array of 64-bit floats
    out_array = weights[0]
    out_array = out_array.flatten("C")
    for i in range(1, len(weights)):
        nparray = weights[i]
        nparray = nparray.flatten("C")
        out_array = np.concatenate((out_array, nparray))
    # Save array to .f64 file
    out_array.tofile(os.path.join("data", file_name))
    ###
    parameters_proto = parameters_to_proto(ins.parameters)
    config_msg = metrics_to_proto(ins.config)
    return ServerMessage.EvaluateIns(parameters=parameters_proto, config=config_msg)


def evaluate_ins_from_proto(msg: ServerMessage.EvaluateIns, name: str) -> typing.EvaluateIns:
    """Deserialize flower.EvaluateIns from ProtoBuf message."""
    parameters = parameters_from_proto(msg.parameters)
    config = metrics_from_proto(msg.config)
    return typing.EvaluateIns(parameters=parameters, config=config)


def evaluate_res_to_proto(res: typing.EvaluateRes, name: str) -> ClientMessage.EvaluateRes:
    """Serialize flower.EvaluateIns to ProtoBuf message."""
    metrics_msg = None if res.metrics is None else metrics_to_proto(res.metrics)
    # Legacy case, will be removed in a future release
    if res.accuracy is not None:
        return ClientMessage.EvaluateRes(
            loss=res.loss,
            num_examples=res.num_examples,
            accuracy=res.accuracy,  # Deprecated
            metrics=metrics_msg,
        )
    # Forward-compatible case
    return ClientMessage.EvaluateRes(
        loss=res.loss,
        num_examples=res.num_examples,
        metrics=metrics_msg,
    )


def evaluate_res_from_proto(msg: ClientMessage.EvaluateRes, cid: str) -> typing.EvaluateRes:
    """Deserialize flower.EvaluateRes from ProtoBuf message."""
    metrics = None if msg.metrics is None else metrics_from_proto(msg.metrics)
    return typing.EvaluateRes(
        loss=msg.loss,
        num_examples=msg.num_examples,
        accuracy=msg.accuracy,  # Deprecated
        metrics=metrics,
    )


# === Metrics messages ===


def metrics_to_proto(metrics: typing.Metrics) -> Any:
    """Serialize... ."""
    proto = {}
    for key in metrics:
        proto[key] = scalar_to_proto(metrics[key])
    return proto


def metrics_from_proto(proto: Any) -> typing.Metrics:
    """Deserialize... ."""
    metrics = {}
    for k in proto:
        metrics[k] = scalar_from_proto(proto[k])
    return metrics


def scalar_to_proto(scalar: typing.Scalar) -> Scalar:
    """Serialize... ."""

    if isinstance(scalar, bool):
        return Scalar(bool=scalar)

    if isinstance(scalar, bytes):
        return Scalar(bytes=scalar)

    if isinstance(scalar, float):
        return Scalar(double=scalar)

    if isinstance(scalar, int):
        return Scalar(sint64=scalar)

    if isinstance(scalar, str):
        return Scalar(string=scalar)

    raise Exception(
        f"Accepted types: {bool, bytes, float, int, str} (but not {type(scalar)})"
    )


def scalar_from_proto(scalar_msg: Scalar) -> typing.Scalar:
    """Deserialize... ."""
    scalar = getattr(scalar_msg, scalar_msg.WhichOneof("scalar"))
    return cast(typing.Scalar, scalar)
