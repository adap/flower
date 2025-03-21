# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
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


from collections import OrderedDict
from collections.abc import MutableMapping
from typing import Any, TypeVar, cast

from google.protobuf.message import Message as GrpcMessage

# pylint: disable=E0611
from flwr.proto.clientappio_pb2 import ClientAppOutputCode, ClientAppOutputStatus
from flwr.proto.error_pb2 import Error as ProtoError
from flwr.proto.fab_pb2 import Fab as ProtoFab
from flwr.proto.message_pb2 import Context as ProtoContext
from flwr.proto.message_pb2 import Message as ProtoMessage
from flwr.proto.message_pb2 import Metadata as ProtoMetadata
from flwr.proto.recorddict_pb2 import Array as ProtoArray
from flwr.proto.recorddict_pb2 import ArrayRecord as ProtoArrayRecord
from flwr.proto.recorddict_pb2 import BoolList, BytesList
from flwr.proto.recorddict_pb2 import ConfigRecord as ProtoConfigRecord
from flwr.proto.recorddict_pb2 import ConfigRecordValue as ProtoConfigRecordValue
from flwr.proto.recorddict_pb2 import DoubleList
from flwr.proto.recorddict_pb2 import MetricRecord as ProtoMetricRecord
from flwr.proto.recorddict_pb2 import MetricRecordValue as ProtoMetricRecordValue
from flwr.proto.recorddict_pb2 import RecordDict as ProtoRecordDict
from flwr.proto.recorddict_pb2 import SintList, StringList, UintList
from flwr.proto.run_pb2 import Run as ProtoRun
from flwr.proto.run_pb2 import RunStatus as ProtoRunStatus
from flwr.proto.transport_pb2 import (
    ClientMessage,
    Code,
    Parameters,
    Reason,
    Scalar,
    ServerMessage,
    Status,
)

# pylint: enable=E0611
from . import (
    Array,
    ArrayRecord,
    ConfigRecord,
    Context,
    MetricRecord,
    RecordDict,
    typing,
)
from .message import Error, Message, Metadata, make_message
from .record.typeddict import TypedDict

#  === Parameters message ===


def parameters_to_proto(parameters: typing.Parameters) -> Parameters:
    """Serialize `Parameters` to ProtoBuf."""
    return Parameters(tensors=parameters.tensors, tensor_type=parameters.tensor_type)


def parameters_from_proto(msg: Parameters) -> typing.Parameters:
    """Deserialize `Parameters` from ProtoBuf."""
    tensors: list[bytes] = list(msg.tensors)
    return typing.Parameters(tensors=tensors, tensor_type=msg.tensor_type)


#  === ReconnectIns message ===


def reconnect_ins_to_proto(ins: typing.ReconnectIns) -> ServerMessage.ReconnectIns:
    """Serialize `ReconnectIns` to ProtoBuf."""
    if ins.seconds is not None:
        return ServerMessage.ReconnectIns(seconds=ins.seconds)
    return ServerMessage.ReconnectIns()


# === DisconnectRes message ===


def disconnect_res_from_proto(msg: ClientMessage.DisconnectRes) -> typing.DisconnectRes:
    """Deserialize `DisconnectRes` from ProtoBuf."""
    if msg.reason == Reason.RECONNECT:
        return typing.DisconnectRes(reason="RECONNECT")
    if msg.reason == Reason.POWER_DISCONNECTED:
        return typing.DisconnectRes(reason="POWER_DISCONNECTED")
    if msg.reason == Reason.WIFI_UNAVAILABLE:
        return typing.DisconnectRes(reason="WIFI_UNAVAILABLE")
    return typing.DisconnectRes(reason="UNKNOWN")


# === GetParameters messages ===


def get_parameters_ins_to_proto(
    ins: typing.GetParametersIns,
) -> ServerMessage.GetParametersIns:
    """Serialize `GetParametersIns` to ProtoBuf."""
    config = properties_to_proto(ins.config)
    return ServerMessage.GetParametersIns(config=config)


def get_parameters_ins_from_proto(
    msg: ServerMessage.GetParametersIns,
) -> typing.GetParametersIns:
    """Deserialize `GetParametersIns` from ProtoBuf."""
    config = properties_from_proto(msg.config)
    return typing.GetParametersIns(config=config)


def get_parameters_res_to_proto(
    res: typing.GetParametersRes,
) -> ClientMessage.GetParametersRes:
    """Serialize `GetParametersRes` to ProtoBuf."""
    status_msg = status_to_proto(res.status)
    if res.status.code == typing.Code.GET_PARAMETERS_NOT_IMPLEMENTED:
        return ClientMessage.GetParametersRes(status=status_msg)
    parameters_proto = parameters_to_proto(res.parameters)
    return ClientMessage.GetParametersRes(
        status=status_msg, parameters=parameters_proto
    )


def get_parameters_res_from_proto(
    msg: ClientMessage.GetParametersRes,
) -> typing.GetParametersRes:
    """Deserialize `GetParametersRes` from ProtoBuf."""
    status = status_from_proto(msg=msg.status)
    parameters = parameters_from_proto(msg.parameters)
    return typing.GetParametersRes(status=status, parameters=parameters)


# === Fit messages ===


def fit_ins_to_proto(ins: typing.FitIns) -> ServerMessage.FitIns:
    """Serialize `FitIns` to ProtoBuf."""
    parameters_proto = parameters_to_proto(ins.parameters)
    config_msg = metrics_to_proto(ins.config)
    return ServerMessage.FitIns(parameters=parameters_proto, config=config_msg)


def fit_ins_from_proto(msg: ServerMessage.FitIns) -> typing.FitIns:
    """Deserialize `FitIns` from ProtoBuf."""
    parameters = parameters_from_proto(msg.parameters)
    config = metrics_from_proto(msg.config)
    return typing.FitIns(parameters=parameters, config=config)


def fit_res_to_proto(res: typing.FitRes) -> ClientMessage.FitRes:
    """Serialize `FitIns` to ProtoBuf."""
    status_msg = status_to_proto(res.status)
    if res.status.code == typing.Code.FIT_NOT_IMPLEMENTED:
        return ClientMessage.FitRes(status=status_msg)
    parameters_proto = parameters_to_proto(res.parameters)
    metrics_msg = None if res.metrics is None else metrics_to_proto(res.metrics)
    return ClientMessage.FitRes(
        status=status_msg,
        parameters=parameters_proto,
        num_examples=res.num_examples,
        metrics=metrics_msg,
    )


def fit_res_from_proto(msg: ClientMessage.FitRes) -> typing.FitRes:
    """Deserialize `FitRes` from ProtoBuf."""
    status = status_from_proto(msg=msg.status)
    parameters = parameters_from_proto(msg.parameters)
    metrics = None if msg.metrics is None else metrics_from_proto(msg.metrics)
    return typing.FitRes(
        status=status,
        parameters=parameters,
        num_examples=msg.num_examples,
        metrics=metrics,
    )


# === GetProperties messages ===


def get_properties_ins_to_proto(
    ins: typing.GetPropertiesIns,
) -> ServerMessage.GetPropertiesIns:
    """Serialize `GetPropertiesIns` to ProtoBuf."""
    config = properties_to_proto(ins.config)
    return ServerMessage.GetPropertiesIns(config=config)


def get_properties_ins_from_proto(
    msg: ServerMessage.GetPropertiesIns,
) -> typing.GetPropertiesIns:
    """Deserialize `GetPropertiesIns` from ProtoBuf."""
    config = properties_from_proto(msg.config)
    return typing.GetPropertiesIns(config=config)


def get_properties_res_to_proto(
    res: typing.GetPropertiesRes,
) -> ClientMessage.GetPropertiesRes:
    """Serialize `GetPropertiesIns` to ProtoBuf."""
    status_msg = status_to_proto(res.status)
    if res.status.code == typing.Code.GET_PROPERTIES_NOT_IMPLEMENTED:
        return ClientMessage.GetPropertiesRes(status=status_msg)
    properties_msg = properties_to_proto(res.properties)
    return ClientMessage.GetPropertiesRes(status=status_msg, properties=properties_msg)


def get_properties_res_from_proto(
    msg: ClientMessage.GetPropertiesRes,
) -> typing.GetPropertiesRes:
    """Deserialize `GetPropertiesRes` from ProtoBuf."""
    status = status_from_proto(msg=msg.status)
    properties = properties_from_proto(msg.properties)
    return typing.GetPropertiesRes(status=status, properties=properties)


# === Evaluate messages ===


def evaluate_ins_to_proto(ins: typing.EvaluateIns) -> ServerMessage.EvaluateIns:
    """Serialize `EvaluateIns` to ProtoBuf."""
    parameters_proto = parameters_to_proto(ins.parameters)
    config_msg = metrics_to_proto(ins.config)
    return ServerMessage.EvaluateIns(parameters=parameters_proto, config=config_msg)


def evaluate_ins_from_proto(msg: ServerMessage.EvaluateIns) -> typing.EvaluateIns:
    """Deserialize `EvaluateIns` from ProtoBuf."""
    parameters = parameters_from_proto(msg.parameters)
    config = metrics_from_proto(msg.config)
    return typing.EvaluateIns(parameters=parameters, config=config)


def evaluate_res_to_proto(res: typing.EvaluateRes) -> ClientMessage.EvaluateRes:
    """Serialize `EvaluateIns` to ProtoBuf."""
    status_msg = status_to_proto(res.status)
    if res.status.code == typing.Code.EVALUATE_NOT_IMPLEMENTED:
        return ClientMessage.EvaluateRes(status=status_msg)
    metrics_msg = None if res.metrics is None else metrics_to_proto(res.metrics)
    return ClientMessage.EvaluateRes(
        status=status_msg,
        loss=res.loss,
        num_examples=res.num_examples,
        metrics=metrics_msg,
    )


def evaluate_res_from_proto(msg: ClientMessage.EvaluateRes) -> typing.EvaluateRes:
    """Deserialize `EvaluateRes` from ProtoBuf."""
    status = status_from_proto(msg=msg.status)
    metrics = None if msg.metrics is None else metrics_from_proto(msg.metrics)
    return typing.EvaluateRes(
        status=status,
        loss=msg.loss,
        num_examples=msg.num_examples,
        metrics=metrics,
    )


# === Status messages ===


def status_to_proto(status: typing.Status) -> Status:
    """Serialize `Status` to ProtoBuf."""
    code = Code.OK
    if status.code == typing.Code.GET_PROPERTIES_NOT_IMPLEMENTED:
        code = Code.GET_PROPERTIES_NOT_IMPLEMENTED
    if status.code == typing.Code.GET_PARAMETERS_NOT_IMPLEMENTED:
        code = Code.GET_PARAMETERS_NOT_IMPLEMENTED
    if status.code == typing.Code.FIT_NOT_IMPLEMENTED:
        code = Code.FIT_NOT_IMPLEMENTED
    if status.code == typing.Code.EVALUATE_NOT_IMPLEMENTED:
        code = Code.EVALUATE_NOT_IMPLEMENTED
    return Status(code=code, message=status.message)


def status_from_proto(msg: Status) -> typing.Status:
    """Deserialize `Status` from ProtoBuf."""
    code = typing.Code.OK
    if msg.code == Code.GET_PROPERTIES_NOT_IMPLEMENTED:
        code = typing.Code.GET_PROPERTIES_NOT_IMPLEMENTED
    if msg.code == Code.GET_PARAMETERS_NOT_IMPLEMENTED:
        code = typing.Code.GET_PARAMETERS_NOT_IMPLEMENTED
    if msg.code == Code.FIT_NOT_IMPLEMENTED:
        code = typing.Code.FIT_NOT_IMPLEMENTED
    if msg.code == Code.EVALUATE_NOT_IMPLEMENTED:
        code = typing.Code.EVALUATE_NOT_IMPLEMENTED
    return typing.Status(code=code, message=msg.message)


# === Properties messages ===


def properties_to_proto(properties: typing.Properties) -> Any:
    """Serialize `Properties` to ProtoBuf."""
    proto = {}
    for key in properties:
        proto[key] = scalar_to_proto(properties[key])
    return proto


def properties_from_proto(proto: Any) -> typing.Properties:
    """Deserialize `Properties` from ProtoBuf."""
    properties = {}
    for k in proto:
        properties[k] = scalar_from_proto(proto[k])
    return properties


# === Metrics messages ===


def metrics_to_proto(metrics: typing.Metrics) -> Any:
    """Serialize `Metrics` to ProtoBuf."""
    proto = {}
    for key in metrics:
        proto[key] = scalar_to_proto(metrics[key])
    return proto


def metrics_from_proto(proto: Any) -> typing.Metrics:
    """Deserialize `Metrics` from ProtoBuf."""
    metrics = {}
    for k in proto:
        metrics[k] = scalar_from_proto(proto[k])
    return metrics


# === Scalar messages ===
INT64_MAX_VALUE = 9223372036854775807  # (1 << 63) - 1


def scalar_to_proto(scalar: typing.Scalar) -> Scalar:
    """Serialize `Scalar` to ProtoBuf."""
    if isinstance(scalar, bool):
        return Scalar(bool=scalar)

    if isinstance(scalar, bytes):
        return Scalar(bytes=scalar)

    if isinstance(scalar, float):
        return Scalar(double=scalar)

    if isinstance(scalar, int):
        # Use uint64 for integers larger than the maximum value of sint64
        if scalar > INT64_MAX_VALUE:
            return Scalar(uint64=scalar)
        return Scalar(sint64=scalar)

    if isinstance(scalar, str):
        return Scalar(string=scalar)

    raise ValueError(
        f"Accepted types: {bool, bytes, float, int, str} (but not {type(scalar)})"
    )


def scalar_from_proto(scalar_msg: Scalar) -> typing.Scalar:
    """Deserialize `Scalar` from ProtoBuf."""
    scalar_field = scalar_msg.WhichOneof("scalar")
    scalar = getattr(scalar_msg, cast(str, scalar_field))
    return cast(typing.Scalar, scalar)


# === Record messages ===


_type_to_field: dict[type, str] = {
    float: "double",
    int: "sint64",
    bool: "bool",
    str: "string",
    bytes: "bytes",
}
_list_type_to_class_and_field: dict[type, tuple[type[GrpcMessage], str]] = {
    float: (DoubleList, "double_list"),
    int: (SintList, "sint_list"),
    bool: (BoolList, "bool_list"),
    str: (StringList, "string_list"),
    bytes: (BytesList, "bytes_list"),
}
T = TypeVar("T")


def _is_uint64(value: Any) -> bool:
    """Check if a value is uint64."""
    return isinstance(value, int) and value > INT64_MAX_VALUE


def _record_value_to_proto(
    value: Any, allowed_types: list[type], proto_class: type[T]
) -> T:
    """Serialize `*RecordValue` to ProtoBuf.

    Note: `bool` MUST be put in the front of allowd_types if it exists.
    """
    arg = {}
    for t in allowed_types:
        # Single element
        # Note: `isinstance(False, int) == True`.
        if isinstance(value, t):
            fld = _type_to_field[t]
            if t is int and _is_uint64(value):
                fld = "uint64"
            arg[fld] = value
            return proto_class(**arg)
        # List
        if isinstance(value, list) and all(isinstance(item, t) for item in value):
            list_class, fld = _list_type_to_class_and_field[t]
            # Use UintList if any element is of type `uint64`.
            if t is int and any(_is_uint64(v) for v in value):
                list_class, fld = UintList, "uint_list"
            arg[fld] = list_class(vals=value)
            return proto_class(**arg)
    # Invalid types
    raise TypeError(
        f"The type of the following value is not allowed "
        f"in '{proto_class.__name__}':\n{value}"
    )


def _record_value_from_proto(value_proto: GrpcMessage) -> Any:
    """Deserialize `*RecordValue` from ProtoBuf."""
    value_field = cast(str, value_proto.WhichOneof("value"))
    if value_field.endswith("list"):
        value = list(getattr(value_proto, value_field).vals)
    else:
        value = getattr(value_proto, value_field)
    return value


def _record_value_dict_to_proto(
    value_dict: TypedDict[str, Any],
    allowed_types: list[type],
    value_proto_class: type[T],
) -> dict[str, T]:
    """Serialize the record value dict to ProtoBuf.

    Note: `bool` MUST be put in the front of allowd_types if it exists.
    """
    # Move bool to the front
    if bool in allowed_types and allowed_types[0] != bool:
        allowed_types.remove(bool)
        allowed_types.insert(0, bool)

    def proto(_v: Any) -> T:
        return _record_value_to_proto(_v, allowed_types, value_proto_class)

    return {k: proto(v) for k, v in value_dict.items()}


def _record_value_dict_from_proto(
    value_dict_proto: MutableMapping[str, Any]
) -> dict[str, Any]:
    """Deserialize the record value dict from ProtoBuf."""
    return {k: _record_value_from_proto(v) for k, v in value_dict_proto.items()}


def array_to_proto(array: Array) -> ProtoArray:
    """Serialize Array to ProtoBuf."""
    return ProtoArray(**vars(array))


def array_from_proto(array_proto: ProtoArray) -> Array:
    """Deserialize Array from ProtoBuf."""
    return Array(
        dtype=array_proto.dtype,
        shape=list(array_proto.shape),
        stype=array_proto.stype,
        data=array_proto.data,
    )


def array_record_to_proto(record: ArrayRecord) -> ProtoArrayRecord:
    """Serialize ArrayRecord to ProtoBuf."""
    return ProtoArrayRecord(
        data_keys=record.keys(),
        data_values=map(array_to_proto, record.values()),
    )


def array_record_from_proto(
    record_proto: ProtoArrayRecord,
) -> ArrayRecord:
    """Deserialize ArrayRecord from ProtoBuf."""
    return ArrayRecord(
        array_dict=OrderedDict(
            zip(record_proto.data_keys, map(array_from_proto, record_proto.data_values))
        ),
        keep_input=False,
    )


def metric_record_to_proto(record: MetricRecord) -> ProtoMetricRecord:
    """Serialize MetricRecord to ProtoBuf."""
    return ProtoMetricRecord(
        data=_record_value_dict_to_proto(record, [float, int], ProtoMetricRecordValue)
    )


def metric_record_from_proto(record_proto: ProtoMetricRecord) -> MetricRecord:
    """Deserialize MetricRecord from ProtoBuf."""
    return MetricRecord(
        metric_dict=cast(
            dict[str, typing.MetricRecordValues],
            _record_value_dict_from_proto(record_proto.data),
        ),
        keep_input=False,
    )


def config_record_to_proto(record: ConfigRecord) -> ProtoConfigRecord:
    """Serialize ConfigRecord to ProtoBuf."""
    return ProtoConfigRecord(
        data=_record_value_dict_to_proto(
            record,
            [bool, int, float, str, bytes],
            ProtoConfigRecordValue,
        )
    )


def config_record_from_proto(record_proto: ProtoConfigRecord) -> ConfigRecord:
    """Deserialize ConfigRecord from ProtoBuf."""
    return ConfigRecord(
        config_dict=cast(
            dict[str, typing.ConfigRecordValues],
            _record_value_dict_from_proto(record_proto.data),
        ),
        keep_input=False,
    )


# === Error message ===


def error_to_proto(error: Error) -> ProtoError:
    """Serialize Error to ProtoBuf."""
    reason = error.reason if error.reason else ""
    return ProtoError(code=error.code, reason=reason)


def error_from_proto(error_proto: ProtoError) -> Error:
    """Deserialize Error from ProtoBuf."""
    reason = error_proto.reason if len(error_proto.reason) > 0 else None
    return Error(code=error_proto.code, reason=reason)


# === RecordDict message ===


def recorddict_to_proto(recorddict: RecordDict) -> ProtoRecordDict:
    """Serialize RecordDict to ProtoBuf."""
    return ProtoRecordDict(
        arrays={
            k: array_record_to_proto(v) for k, v in recorddict.array_records.items()
        },
        metrics={
            k: metric_record_to_proto(v) for k, v in recorddict.metric_records.items()
        },
        configs={
            k: config_record_to_proto(v) for k, v in recorddict.config_records.items()
        },
    )


def recorddict_from_proto(recorddict_proto: ProtoRecordDict) -> RecordDict:
    """Deserialize RecordDict from ProtoBuf."""
    ret = RecordDict()
    for k, arr_record_proto in recorddict_proto.arrays.items():
        ret[k] = array_record_from_proto(arr_record_proto)
    for k, m_record_proto in recorddict_proto.metrics.items():
        ret[k] = metric_record_from_proto(m_record_proto)
    for k, c_record_proto in recorddict_proto.configs.items():
        ret[k] = config_record_from_proto(c_record_proto)
    return ret


# === FAB ===


def fab_to_proto(fab: typing.Fab) -> ProtoFab:
    """Create a proto Fab object from a Python Fab."""
    return ProtoFab(hash_str=fab.hash_str, content=fab.content)


def fab_from_proto(fab: ProtoFab) -> typing.Fab:
    """Create a Python Fab object from a proto Fab."""
    return typing.Fab(fab.hash_str, fab.content)


# === User configs ===


def user_config_to_proto(user_config: typing.UserConfig) -> Any:
    """Serialize `UserConfig` to ProtoBuf."""
    proto = {}
    for key, value in user_config.items():
        proto[key] = user_config_value_to_proto(value)
    return proto


def user_config_from_proto(proto: Any) -> typing.UserConfig:
    """Deserialize `UserConfig` from ProtoBuf."""
    metrics = {}
    for key, value in proto.items():
        metrics[key] = user_config_value_from_proto(value)
    return metrics


def user_config_value_to_proto(user_config_value: typing.UserConfigValue) -> Scalar:
    """Serialize `UserConfigValue` to ProtoBuf."""
    if isinstance(user_config_value, bool):
        return Scalar(bool=user_config_value)

    if isinstance(user_config_value, float):
        return Scalar(double=user_config_value)

    if isinstance(user_config_value, int):
        return Scalar(sint64=user_config_value)

    if isinstance(user_config_value, str):
        return Scalar(string=user_config_value)

    raise ValueError(
        f"Accepted types: {bool, float, int, str} (but not {type(user_config_value)})"
    )


def user_config_value_from_proto(scalar_msg: Scalar) -> typing.UserConfigValue:
    """Deserialize `UserConfigValue` from ProtoBuf."""
    scalar_field = scalar_msg.WhichOneof("scalar")
    scalar = getattr(scalar_msg, cast(str, scalar_field))
    return cast(typing.UserConfigValue, scalar)


# === Metadata messages ===


def metadata_to_proto(metadata: Metadata) -> ProtoMetadata:
    """Serialize `Metadata` to ProtoBuf."""
    proto = ProtoMetadata(  # pylint: disable=E1101
        run_id=metadata.run_id,
        message_id=metadata.message_id,
        src_node_id=metadata.src_node_id,
        dst_node_id=metadata.dst_node_id,
        reply_to_message_id=metadata.reply_to_message_id,
        group_id=metadata.group_id,
        ttl=metadata.ttl,
        message_type=metadata.message_type,
        created_at=metadata.created_at,
    )
    return proto


def metadata_from_proto(metadata_proto: ProtoMetadata) -> Metadata:
    """Deserialize `Metadata` from ProtoBuf."""
    metadata = Metadata(
        run_id=metadata_proto.run_id,
        message_id=metadata_proto.message_id,
        src_node_id=metadata_proto.src_node_id,
        dst_node_id=metadata_proto.dst_node_id,
        reply_to_message_id=metadata_proto.reply_to_message_id,
        group_id=metadata_proto.group_id,
        created_at=metadata_proto.created_at,
        ttl=metadata_proto.ttl,
        message_type=metadata_proto.message_type,
    )
    return metadata


# === Message messages ===


def message_to_proto(message: Message) -> ProtoMessage:
    """Serialize `Message` to ProtoBuf."""
    proto = ProtoMessage(
        metadata=metadata_to_proto(message.metadata),
        content=(
            recorddict_to_proto(message.content) if message.has_content() else None
        ),
        error=error_to_proto(message.error) if message.has_error() else None,
    )
    return proto


def message_from_proto(message_proto: ProtoMessage) -> Message:
    """Deserialize `Message` from ProtoBuf."""
    return make_message(
        metadata=metadata_from_proto(message_proto.metadata),
        content=(
            recorddict_from_proto(message_proto.content)
            if message_proto.HasField("content")
            else None
        ),
        error=(
            error_from_proto(message_proto.error)
            if message_proto.HasField("error")
            else None
        ),
    )


# === Context messages ===


def context_to_proto(context: Context) -> ProtoContext:
    """Serialize `Context` to ProtoBuf."""
    proto = ProtoContext(
        run_id=context.run_id,
        node_id=context.node_id,
        node_config=user_config_to_proto(context.node_config),
        state=recorddict_to_proto(context.state),
        run_config=user_config_to_proto(context.run_config),
    )
    return proto


def context_from_proto(context_proto: ProtoContext) -> Context:
    """Deserialize `Context` from ProtoBuf."""
    context = Context(
        run_id=context_proto.run_id,
        node_id=context_proto.node_id,
        node_config=user_config_from_proto(context_proto.node_config),
        state=recorddict_from_proto(context_proto.state),
        run_config=user_config_from_proto(context_proto.run_config),
    )
    return context


# === Run messages ===


def run_to_proto(run: typing.Run) -> ProtoRun:
    """Serialize `Run` to ProtoBuf."""
    proto = ProtoRun(
        run_id=run.run_id,
        fab_id=run.fab_id,
        fab_version=run.fab_version,
        fab_hash=run.fab_hash,
        override_config=user_config_to_proto(run.override_config),
        pending_at=run.pending_at,
        starting_at=run.starting_at,
        running_at=run.running_at,
        finished_at=run.finished_at,
        status=run_status_to_proto(run.status),
    )
    return proto


def run_from_proto(run_proto: ProtoRun) -> typing.Run:
    """Deserialize `Run` from ProtoBuf."""
    run = typing.Run(
        run_id=run_proto.run_id,
        fab_id=run_proto.fab_id,
        fab_version=run_proto.fab_version,
        fab_hash=run_proto.fab_hash,
        override_config=user_config_from_proto(run_proto.override_config),
        pending_at=run_proto.pending_at,
        starting_at=run_proto.starting_at,
        running_at=run_proto.running_at,
        finished_at=run_proto.finished_at,
        status=run_status_from_proto(run_proto.status),
    )
    return run


# === ClientApp status messages ===


def clientappstatus_to_proto(
    status: typing.ClientAppOutputStatus,
) -> ClientAppOutputStatus:
    """Serialize `ClientAppOutputStatus` to ProtoBuf."""
    code = ClientAppOutputCode.SUCCESS
    if status.code == typing.ClientAppOutputCode.DEADLINE_EXCEEDED:
        code = ClientAppOutputCode.DEADLINE_EXCEEDED
    if status.code == typing.ClientAppOutputCode.UNKNOWN_ERROR:
        code = ClientAppOutputCode.UNKNOWN_ERROR
    return ClientAppOutputStatus(code=code, message=status.message)


def clientappstatus_from_proto(
    msg: ClientAppOutputStatus,
) -> typing.ClientAppOutputStatus:
    """Deserialize `ClientAppOutputStatus` from ProtoBuf."""
    code = typing.ClientAppOutputCode.SUCCESS
    if msg.code == ClientAppOutputCode.DEADLINE_EXCEEDED:
        code = typing.ClientAppOutputCode.DEADLINE_EXCEEDED
    if msg.code == ClientAppOutputCode.UNKNOWN_ERROR:
        code = typing.ClientAppOutputCode.UNKNOWN_ERROR
    return typing.ClientAppOutputStatus(code=code, message=msg.message)


# === Run status ===


def run_status_to_proto(run_status: typing.RunStatus) -> ProtoRunStatus:
    """Serialize `RunStatus` to ProtoBuf."""
    return ProtoRunStatus(
        status=run_status.status,
        sub_status=run_status.sub_status,
        details=run_status.details,
    )


def run_status_from_proto(run_status_proto: ProtoRunStatus) -> typing.RunStatus:
    """Deserialize `RunStatus` from ProtoBuf."""
    return typing.RunStatus(
        status=run_status_proto.status,
        sub_status=run_status_proto.sub_status,
        details=run_status_proto.details,
    )
