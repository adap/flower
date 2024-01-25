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
"""(De-)serialization tests."""


from typing import Dict, OrderedDict, Union, cast

# pylint: disable=E0611
from flwr.proto import transport_pb2 as pb2
from flwr.proto.recordset_pb2 import Array as ProtoArray
from flwr.proto.recordset_pb2 import ConfigsRecord as ProtoConfigsRecord
from flwr.proto.recordset_pb2 import MetricsRecord as ProtoMetricsRecord
from flwr.proto.recordset_pb2 import ParametersRecord as ProtoParametersRecord
from flwr.proto.recordset_pb2 import RecordSet as ProtoRecordSet

# pylint: enable=E0611
from . import typing
from .configsrecord import ConfigsRecord
from .metricsrecord import MetricsRecord
from .parametersrecord import Array, ParametersRecord
from .recordset import RecordSet
from .serde import (
    array_from_proto,
    array_to_proto,
    configs_record_from_proto,
    configs_record_to_proto,
    metrics_record_from_proto,
    metrics_record_to_proto,
    named_values_from_proto,
    named_values_to_proto,
    parameters_record_from_proto,
    parameters_record_to_proto,
    recordset_from_proto,
    recordset_to_proto,
    scalar_from_proto,
    scalar_to_proto,
    status_from_proto,
    status_to_proto,
    value_from_proto,
    value_to_proto,
)


def test_serialisation_deserialisation() -> None:
    """Test if the np.ndarray is identical after (de-)serialization."""
    # Prepare
    scalars = [True, b"bytestr", 3.14, 9000, "Hello"]

    for scalar in scalars:
        # Execute
        scalar = cast(Union[bool, bytes, float, int, str], scalar)
        serialized = scalar_to_proto(scalar)
        actual = scalar_from_proto(serialized)

        # Assert
        assert actual == scalar


def test_status_to_proto() -> None:
    """Test status message (de-)serialization."""
    # Prepare
    code_msg = pb2.Code.OK  # pylint: disable=E1101
    status_msg = pb2.Status(code=code_msg, message="Success")  # pylint: disable=E1101

    code = typing.Code.OK
    status = typing.Status(code=code, message="Success")

    # Execute
    actual_status_msg = status_to_proto(status=status)

    # Assert
    assert actual_status_msg == status_msg


def test_status_from_proto() -> None:
    """Test status message (de-)serialization."""
    # Prepare
    code_msg = pb2.Code.OK  # pylint: disable=E1101
    status_msg = pb2.Status(code=code_msg, message="Success")  # pylint: disable=E1101

    code = typing.Code.OK
    status = typing.Status(code=code, message="Success")

    # Execute
    actual_status = status_from_proto(msg=status_msg)

    # Assert
    assert actual_status == status


def test_value_serialization_deserialization() -> None:
    """Test if values are identical after (de-)serialization."""
    # Prepare
    values = [
        # boolean scalar and list
        True,
        [True, False, False, True],
        # bytes scalar and list
        b"test \x01\x02\x03 !@#$%^&*()",
        [b"\x0a\x0b", b"\x0c\x0d\x0e", b"\x0f"],
        # float scalar and list
        3.14,
        [2.714, -0.012],
        # integer scalar and list
        23,
        [123456],
        # string scalar and list
        "abcdefghijklmnopqrstuvwxy",
        ["456hgdhfd", "1234567890123456789012345678901", "I'm a string."],
        # empty list
        [],
    ]

    for value in values:
        # Execute
        serialized = value_to_proto(cast(typing.Value, value))
        deserialized = value_from_proto(serialized)

        # Assert
        if isinstance(value, list):
            assert isinstance(deserialized, list)
            assert len(value) == len(deserialized)
            for elm1, elm2 in zip(value, deserialized):
                assert elm1 == elm2
        else:
            assert value == deserialized


def test_named_values_serialization_deserialization() -> None:
    """Test if named values is identical after (de-)serialization."""
    # Prepare
    values = [
        # boolean scalar and list
        True,
        [True, False, False, True],
        # bytes scalar and list
        b"test \x01\x02\x03 !@#$%^&*()",
        [b"\x0a\x0b", b"\x0c\x0d\x0e", b"\x0f"],
        # float scalar and list
        3.14,
        [2.714, -0.012],
        # integer scalar and list
        23,
        [123456],
        # string scalar and list
        "abcdefghijklmnopqrstuvwxy",
        ["456hgdhfd", "1234567890123456789012345678901", "I'm a string."],
        # empty list
        [],
    ]
    named_values = {f"value {i}": value for i, value in enumerate(values)}

    # Execute
    serialized = named_values_to_proto(cast(Dict[str, typing.Value], named_values))
    deserialized = named_values_from_proto(serialized)

    # Assert
    assert len(named_values) == len(deserialized)
    for name in named_values:
        expected = named_values[name]
        actual = deserialized[name]
        if isinstance(expected, list):
            assert isinstance(actual, list)
            assert len(expected) == len(actual)
            for elm1, elm2 in zip(expected, actual):
                assert elm1 == elm2
        else:
            assert expected == actual


def test_array_serialization_deserialization() -> None:
    """Test serialization and deserialization of Array."""
    # Prepare
    original = Array(dtype="float", shape=[2, 2], stype="dense", data=b"1234")

    # Execute
    proto = array_to_proto(original)
    deserialized = array_from_proto(proto)

    # Assert
    assert isinstance(proto, ProtoArray)
    assert original == deserialized


def test_parameters_record_serialization_deserialization() -> None:
    """Test serialization and deserialization of ParametersRecord."""
    # Prepare
    original = ParametersRecord(
        array_dict=OrderedDict(
            [
                ("k1", Array(dtype="float", shape=[2, 2], stype="dense", data=b"1234")),
                ("k2", Array(dtype="int", shape=[3], stype="sparse", data=b"567")),
            ]
        ),
        keep_input=False,
    )

    # Execute
    proto = parameters_record_to_proto(original)
    deserialized = parameters_record_from_proto(proto)

    # Assert
    assert isinstance(proto, ProtoParametersRecord)
    assert original.data == deserialized.data


def test_metrics_record_serialization_deserialization() -> None:
    """Test serialization and deserialization of MetricsRecord."""
    # Prepare
    original = MetricsRecord(
        metrics_dict={"accuracy": 0.95, "loss": 0.1}, keep_input=False
    )

    # Execute
    proto = metrics_record_to_proto(original)
    deserialized = metrics_record_from_proto(proto)

    # Assert
    assert isinstance(proto, ProtoMetricsRecord)
    assert original.data == deserialized.data


def test_configs_record_serialization_deserialization() -> None:
    """Test serialization and deserialization of ConfigsRecord."""
    # Prepare
    original = ConfigsRecord(
        configs_dict={"learning_rate": 0.01, "batch_size": 32}, keep_input=False
    )

    # Execute
    proto = configs_record_to_proto(original)
    deserialized = configs_record_from_proto(proto)

    # Assert
    assert isinstance(proto, ProtoConfigsRecord)
    assert original.data == deserialized.data


def test_recordset_serialization_deserialization() -> None:
    """Test serialization and deserialization of RecordSet."""
    # Prepare
    encoder_params_record = ParametersRecord(
        array_dict=OrderedDict(
            [
                (
                    "k1",
                    Array(dtype="float", shape=[2, 2], stype="dense", data=b"1234"),
                ),
                ("k2", Array(dtype="int", shape=[3], stype="sparse", data=b"567")),
            ]
        ),
        keep_input=False,
    )
    decoder_params_record = ParametersRecord(
        array_dict=OrderedDict(
            [
                (
                    "k1",
                    Array(
                        dtype="float", shape=[32, 32, 4], stype="dense", data=b"0987"
                    ),
                ),
            ]
        ),
        keep_input=False,
    )

    original = RecordSet(
        parameters={
            "encoder_parameters": encoder_params_record,
            "decoder_parameters": decoder_params_record,
        },
        metrics={
            "acc_metrics": MetricsRecord(
                metrics_dict={"accuracy": 0.95, "loss": 0.1}, keep_input=False
            )
        },
        configs={
            "my_configs": ConfigsRecord(
                configs_dict={
                    "learning_rate": 0.01,
                    "batch_size": 32,
                    "public_key": b"21f8sioj@!#",
                    "log": "Hello, world!",
                },
                keep_input=False,
            )
        },
    )

    # Execute
    proto = recordset_to_proto(original)
    deserialized = recordset_from_proto(proto)

    # Assert
    assert isinstance(proto, ProtoRecordSet)
    assert original == deserialized
