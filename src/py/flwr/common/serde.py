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

from flwr.server.server import Server
from flwr_experimental.baseline import config

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
    return typing.ParametersRes(parameters=parameters)


# === Fit messages ===


def fit_ins_to_proto(ins: typing.FitIns) -> ServerMessage.FitIns:
    """Serialize flower.FitIns to ProtoBuf message."""
    parameters_proto = parameters_to_proto(ins.parameters)
    config_msg = metrics_to_proto(ins.config)
    return ServerMessage.FitIns(parameters=parameters_proto, config=config_msg)


def fit_ins_from_proto(msg: ServerMessage.FitIns) -> typing.FitIns:
    """Deserialize flower.FitIns from ProtoBuf message."""
    parameters = parameters_from_proto(msg.parameters)
    config = metrics_from_proto(msg.config)
    return typing.FitIns(parameters=parameters, config=config)


def fit_res_to_proto(res: typing.FitRes) -> ClientMessage.FitRes:
    """Serialize flower.FitIns to ProtoBuf message."""
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


# === SecAgg Messages ===
# === Check if error ===
def check_error(msg: ClientMessage.SecAggRes):
    if msg.HasField("error_res"):
        raise Exception(msg.error_res.error)


# === Setup Param ===
def setup_param_ins_to_proto(
    setup_param_ins: typing.SetupParamIns,
) -> ServerMessage.SecAggMsg:
    return ServerMessage.SecAggMsg(
        setup_param=ServerMessage.SecAggMsg.SetupParam(
            secagg_id=setup_param_ins.secagg_id,
            sample_num=setup_param_ins.sample_num,
            share_num=setup_param_ins.share_num,
            threshold=setup_param_ins.threshold,
            clipping_range=setup_param_ins.clipping_range,
            target_range=setup_param_ins.target_range,
            mod_range=setup_param_ins.mod_range
        )
    )


def setup_param_from_proto(
    setup_param_msg: ServerMessage.SecAggMsg,
) -> typing.SetupParamIns:
    return typing.SetupParamIns(
        secagg_id=setup_param_msg.setup_param.secagg_id,
        sample_num=setup_param_msg.setup_param.sample_num,
        share_num=setup_param_msg.setup_param.share_num,
        threshold=setup_param_msg.setup_param.threshold,
        clipping_range=setup_param_msg.setup_param.clipping_range,
        target_range=setup_param_msg.setup_param.target_range,
        mod_range=setup_param_msg.setup_param.mod_range,
    )


# === Ask Keys ===
def ask_keys_to_proto() -> ServerMessage.SecAggMsg:
    return ServerMessage.SecAggMsg(ask_keys=ServerMessage.SecAggMsg.AskKeys())


def ask_keys_res_to_proto(res: typing.ParametersRes) -> ClientMessage.SecAggRes:
    return ClientMessage.SecAggRes(
        ask_keys_res=ClientMessage.SecAggRes.AskKeysRes(pk1=res.pk1, pk2=res.pk2)
    )


def ask_keys_res_from_proto(msg: ClientMessage.SecAggRes) -> typing.AskKeysRes:
    return typing.AskKeysRes(pk1=msg.ask_keys_res.pk1, pk2=msg.ask_keys_res.pk2)


# === Share Keys ===
def share_keys_ins_to_proto(share_keys_ins: typing.ShareKeysIns) -> ServerMessage.SecAggMsg:
    public_keys_dict = share_keys_ins.public_keys_dict
    proto_public_keys_dict = {}
    for i in public_keys_dict.keys():
        proto_public_keys_dict[i] = ServerMessage.SecAggMsg.ShareKeys.KeysPair(
            pk1=public_keys_dict[i].pk1, pk2=public_keys_dict[i].pk2
        )
    return ServerMessage.SecAggMsg(
        share_keys=ServerMessage.SecAggMsg.ShareKeys(
            public_keys_dict=proto_public_keys_dict
        )
    )


def share_keys_ins_from_proto(share_keys_msg: ServerMessage.SecAggMsg) -> typing.ShareKeysIns:
    proto_public_keys_dict = share_keys_msg.share_keys.public_keys_dict
    public_keys_dict = {}
    for i in proto_public_keys_dict.keys():
        public_keys_dict[i] = typing.AskKeysRes(
            pk1=proto_public_keys_dict[i].pk1, pk2=proto_public_keys_dict[i].pk2)
    return typing.ShareKeysIns(public_keys_dict=public_keys_dict)


def share_keys_res_to_proto(share_keys_res: typing.ShareKeysRes) -> ClientMessage.SecAggRes:
    share_keys_res_msg = ClientMessage.SecAggRes.ShareKeysRes()
    for packet in share_keys_res.share_keys_res_list:
        proto_packet = ClientMessage.SecAggRes.ShareKeysRes.Packet(
            source=packet.source, destination=packet.destination, ciphertext=packet.ciphertext
        )
        share_keys_res_msg.packet_list.append(proto_packet)
    return ClientMessage.SecAggRes(share_keys_res=share_keys_res_msg)


def share_keys_res_from_proto(share_keys_res_msg: ClientMessage.SecAggRes) -> typing.ShareKeysRes:
    proto_packet_list = share_keys_res_msg.share_keys_res.packet_list
    packet_list = []
    for proto_packet in proto_packet_list:
        packet = typing.ShareKeysPacket(
            source=proto_packet.source, destination=proto_packet.destination, ciphertext=proto_packet.ciphertext
        )
        packet_list.append(packet)
    return typing.ShareKeysRes(share_keys_res_list=packet_list)

# === Ask vectors ===


def ask_vectors_ins_to_proto(ask_vectors_ins: typing.AskVectorsIns) -> ServerMessage.SecAggMsg:
    packet_list = ask_vectors_ins.ask_vectors_in_list
    proto_packet_list = []
    for packet in packet_list:
        proto_packet = ServerMessage.SecAggMsg.AskVectors.Packet(
            source=packet.source, destination=packet.destination, ciphertext=packet.ciphertext)
        proto_packet_list.append(proto_packet)
    fit_ins = ServerMessage.SecAggMsg.AskVectors.FitIns(parameters=parameters_to_proto(
        ask_vectors_ins.fit_ins.parameters), config=metrics_to_proto(ask_vectors_ins.fit_ins.config))
    return ServerMessage.SecAggMsg(ask_vectors=ServerMessage.SecAggMsg.AskVectors(packet_list=proto_packet_list, fit_ins=fit_ins))


def ask_vectors_ins_from_proto(ask_vectors_msg: ServerMessage.SecAggMsg) -> typing.AskVectorsIns:
    proto_packet_list = ask_vectors_msg.ask_vectors.packet_list
    packet_list = []
    for proto_packet in proto_packet_list:
        packet = typing.ShareKeysPacket(
            source=proto_packet.source, destination=proto_packet.destination, ciphertext=proto_packet.ciphertext)
        packet_list.append(packet)
    fit_ins = typing.FitIns(parameters=parameters_from_proto(
        ask_vectors_msg.ask_vectors.fit_ins.parameters), config=metrics_from_proto(ask_vectors_msg.ask_vectors.fit_ins.config))
    return typing.AskVectorsIns(ask_vectors_in_list=packet_list, fit_ins=fit_ins)


def ask_vectors_res_to_proto(ask_vectors_res: typing.AskVectorsRes) -> ClientMessage.SecAggRes:
    parameters_proto = parameters_to_proto(ask_vectors_res.parameters)
    return ClientMessage.SecAggRes(ask_vectors_res=ClientMessage.SecAggRes.AskVectorsRes(parameters=parameters_proto))


def ask_vectors_res_from_proto(ask_vectors_res_msg: ClientMessage.SecAggRes) -> typing.AskVectorsRes:
    parameters = parameters_from_proto(ask_vectors_res_msg.ask_vectors_res.parameters)
    return typing.AskVectorsRes(parameters=parameters)

# === Evaluate messages ===


def evaluate_ins_to_proto(ins: typing.EvaluateIns) -> ServerMessage.EvaluateIns:
    """Serialize flower.EvaluateIns to ProtoBuf message."""
    parameters_proto = parameters_to_proto(ins.parameters)
    config_msg = metrics_to_proto(ins.config)
    return ServerMessage.EvaluateIns(parameters=parameters_proto, config=config_msg)


def evaluate_ins_from_proto(msg: ServerMessage.EvaluateIns) -> typing.EvaluateIns:
    """Deserialize flower.EvaluateIns from ProtoBuf message."""
    parameters = parameters_from_proto(msg.parameters)
    config = metrics_from_proto(msg.config)
    return typing.EvaluateIns(parameters=parameters, config=config)


def evaluate_res_to_proto(res: typing.EvaluateRes) -> ClientMessage.EvaluateRes:
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


def evaluate_res_from_proto(msg: ClientMessage.EvaluateRes) -> typing.EvaluateRes:
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
