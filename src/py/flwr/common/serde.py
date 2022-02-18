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
    Code,
    Parameters,
    Reason,
    Scalar,
    ServerMessage,
    Status,
)

from . import typing

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
    """Serialize Reconnect to ProtoBuf message."""
    if reconnect.seconds is not None:
        return ServerMessage.Reconnect(seconds=reconnect.seconds)
    return ServerMessage.Reconnect()


def reconnect_from_proto(msg: ServerMessage.Reconnect) -> typing.Reconnect:
    """Deserialize Reconnect from ProtoBuf message."""
    return typing.Reconnect(seconds=msg.seconds)


# === Disconnect message ===


def disconnect_to_proto(disconnect: typing.Disconnect) -> ClientMessage.Disconnect:
    """Serialize Disconnect to ProtoBuf message."""
    reason_proto = Reason.UNKNOWN
    if disconnect.reason == "RECONNECT":
        reason_proto = Reason.RECONNECT
    elif disconnect.reason == "POWER_DISCONNECTED":
        reason_proto = Reason.POWER_DISCONNECTED
    elif disconnect.reason == "WIFI_UNAVAILABLE":
        reason_proto = Reason.WIFI_UNAVAILABLE
    return ClientMessage.Disconnect(reason=reason_proto)


def disconnect_from_proto(msg: ClientMessage.Disconnect) -> typing.Disconnect:
    """Deserialize Disconnect from ProtoBuf message."""
    if msg.reason == Reason.RECONNECT:
        return typing.Disconnect(reason="RECONNECT")
    if msg.reason == Reason.POWER_DISCONNECTED:
        return typing.Disconnect(reason="POWER_DISCONNECTED")
    if msg.reason == Reason.WIFI_UNAVAILABLE:
        return typing.Disconnect(reason="WIFI_UNAVAILABLE")
    return typing.Disconnect(reason="UNKNOWN")


# === GetParameters messages ===


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
    """Serialize FitIns to ProtoBuf message."""
    parameters_proto = parameters_to_proto(ins.parameters)
    config_msg = metrics_to_proto(ins.config)
    return ServerMessage.FitIns(parameters=parameters_proto, config=config_msg)


def fit_ins_from_proto(msg: ServerMessage.FitIns) -> typing.FitIns:
    """Deserialize FitIns from ProtoBuf message."""
    parameters = parameters_from_proto(msg.parameters)
    config = metrics_from_proto(msg.config)
    return typing.FitIns(parameters=parameters, config=config)


def fit_res_to_proto(res: typing.FitRes) -> ClientMessage.FitRes:
    """Serialize FitIns to ProtoBuf message."""
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


def fit_res_from_proto(msg: ClientMessage.FitRes) -> typing.FitRes:
    """Deserialize FitRes from ProtoBuf message."""
    parameters = parameters_from_proto(msg.parameters)
    metrics = None if msg.metrics is None else metrics_from_proto(msg.metrics)
    return typing.FitRes(
        parameters=parameters,
        num_examples=msg.num_examples,
        num_examples_ceil=msg.num_examples_ceil,  # Deprecated
        fit_duration=msg.fit_duration,  # Deprecated
        metrics=metrics,
    )


# === Properties messages ===


def properties_ins_to_proto(ins: typing.PropertiesIns) -> ServerMessage.PropertiesIns:
    """Serialize PropertiesIns to ProtoBuf message."""
    config = properties_to_proto(ins.config)
    return ServerMessage.PropertiesIns(config=config)


def properties_ins_from_proto(msg: ServerMessage.PropertiesIns) -> typing.PropertiesIns:
    """Deserialize PropertiesIns from ProtoBuf message."""
    config = properties_from_proto(msg.config)
    return typing.PropertiesIns(config=config)


def properties_res_to_proto(res: typing.PropertiesRes) -> ClientMessage.PropertiesRes:
    """Serialize PropertiesIns to ProtoBuf message."""
    status_msg = status_to_proto(res.status)
    properties_msg = properties_to_proto(res.properties)
    return ClientMessage.PropertiesRes(status=status_msg, properties=properties_msg)


def properties_res_from_proto(msg: ClientMessage.PropertiesRes) -> typing.PropertiesRes:
    """Deserialize PropertiesRes from ProtoBuf message."""
    status = status_from_proto(msg=msg.status)
    properties = properties_from_proto(msg.properties)
    return typing.PropertiesRes(status=status, properties=properties)


def status_to_proto(status: typing.Status) -> Status:
    """Serialize Code to ProtoBuf message."""
    code = Code.OK
    if status.code == typing.Code.GET_PARAMETERS_NOT_IMPLEMENTED:
        code = Code.GET_PARAMETERS_NOT_IMPLEMENTED
    return Status(code=code, message=status.message)


def status_from_proto(msg: Status) -> typing.Status:
    """Deserialize Code from ProtoBuf message."""
    code = typing.Code.OK
    if msg.code == Code.GET_PARAMETERS_NOT_IMPLEMENTED:
        code = typing.Code.GET_PARAMETERS_NOT_IMPLEMENTED
    return typing.Status(code=code, message=msg.message)


# === Evaluate messages ===


def evaluate_ins_to_proto(ins: typing.EvaluateIns) -> ServerMessage.EvaluateIns:
    """Serialize EvaluateIns to ProtoBuf message."""
    parameters_proto = parameters_to_proto(ins.parameters)
    config_msg = metrics_to_proto(ins.config)
    return ServerMessage.EvaluateIns(parameters=parameters_proto, config=config_msg)


def evaluate_ins_from_proto(msg: ServerMessage.EvaluateIns) -> typing.EvaluateIns:
    """Deserialize EvaluateIns from ProtoBuf message."""
    parameters = parameters_from_proto(msg.parameters)
    config = metrics_from_proto(msg.config)
    return typing.EvaluateIns(parameters=parameters, config=config)


def evaluate_res_to_proto(res: typing.EvaluateRes) -> ClientMessage.EvaluateRes:
    """Serialize EvaluateIns to ProtoBuf message."""
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


def evaluate_res_from_proto(msg: ClientMessage.EvaluateRes) -> typing.EvaluateRes:
    """Deserialize EvaluateRes from ProtoBuf message."""
    metrics = None if msg.metrics is None else metrics_from_proto(msg.metrics)
    return typing.EvaluateRes(
        loss=msg.loss,
        num_examples=msg.num_examples,
        accuracy=msg.accuracy,  # Deprecated
        metrics=metrics,
    )


# === Properties messages ===


def properties_to_proto(properties: typing.Properties) -> Any:
    """Serialize... ."""
    proto = {}
    for key in properties:
        proto[key] = scalar_to_proto(properties[key])
    return proto


def properties_from_proto(proto: Any) -> typing.Properties:
    """Deserialize... ."""
    properties = {}
    for k in proto:
        properties[k] = scalar_from_proto(proto[k])
    return properties


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


# === Scalar messages ===


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
    scalar_field = scalar_msg.WhichOneof("scalar")
    scalar = getattr(scalar_msg, cast(str, scalar_field))
    return cast(typing.Scalar, scalar)
