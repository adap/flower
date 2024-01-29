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


import random
import string
from typing import Any, Dict, Optional, OrderedDict, Type, TypeVar, Union, cast

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
from .message import Message, Metadata
from .metricsrecord import MetricsRecord
from .parametersrecord import Array, ParametersRecord
from .recordset import RecordSet
from .serde import (
    array_from_proto,
    array_to_proto,
    configs_record_from_proto,
    configs_record_to_proto,
    message_from_taskins,
    message_from_taskres,
    message_to_taskins,
    message_to_taskres,
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


T = TypeVar("T")


class RecordMaker:
    """A record maker based on a seeded random number generator."""

    def __init__(self, state: int = 42) -> None:
        self.rng = random.Random(state)

    def randbytes(self, n: int) -> bytes:
        """Create a bytes."""
        return self.rng.getrandbits(n * 8).to_bytes(n, "little")

    def get_str(self, length: Optional[int] = None) -> str:
        """Create a string."""
        char_pool = (
            string.ascii_letters + string.digits + " !@#$%^&*()_-+=[]|;':,./<>?{}"
        )
        if length is None:
            length = self.rng.randint(1, 10)
        return "".join(self.rng.choices(char_pool, k=length))

    def get_value(self, dtype: Type[T]) -> T:
        """Create a value of a given type."""
        ret: Any = None
        if dtype == bool:
            ret = self.rng.random() < 0.5
        elif dtype == str:
            ret = self.get_str(self.rng.randint(10, 100))
        elif dtype == int:
            ret = self.rng.randint(-1 << 30, 1 << 30)
        elif dtype == float:
            ret = (self.rng.random() - 0.5) * (2.0 ** self.rng.randint(0, 50))
        elif dtype == bytes:
            ret = self.randbytes(self.rng.randint(10, 100))
        else:
            raise NotImplementedError(f"Unsupported dtype: {dtype}")
        return cast(T, ret)

    def array(self) -> Array:
        """Create a Array."""
        dtypes = ("float", "int")
        stypes = ("torch", "tf", "numpy")
        max_shape_size = 100
        max_shape_dim = 10
        min_max_bytes_size = (10, 1000)

        dtype = self.rng.choice(dtypes)
        shape = [
            self.rng.randint(1, max_shape_size)
            for _ in range(self.rng.randint(1, max_shape_dim))
        ]
        stype = self.rng.choice(stypes)
        data = self.randbytes(self.rng.randint(*min_max_bytes_size))
        return Array(dtype=dtype, shape=shape, stype=stype, data=data)

    def parameters_record(self) -> ParametersRecord:
        """Create a ParametersRecord."""
        num_arrays = self.rng.randint(1, 5)
        arrays = OrderedDict(
            [(self.get_str(), self.array()) for i in range(num_arrays)]
        )
        return ParametersRecord(arrays, keep_input=False)

    def metrics_record(self) -> MetricsRecord:
        """Create a MetricsRecord."""
        num_entries = self.rng.randint(1, 5)
        types = (float, int)
        return MetricsRecord(
            metrics_dict={
                self.get_str(): self.get_value(self.rng.choice(types))
                for _ in range(num_entries)
            },
            keep_input=False,
        )

    def configs_record(self) -> ConfigsRecord:
        """Create a ConfigsRecord."""
        num_entries = self.rng.randint(1, 5)
        types = (str, int, float, bytes, bool)
        return ConfigsRecord(
            configs_dict={
                self.get_str(): self.get_value(self.rng.choice(types))
                for _ in range(num_entries)
            },
            keep_input=False,
        )

    def recordset(
        self,
        num_params_records: int,
        num_metrics_records: int,
        num_configs_records: int,
    ) -> RecordSet:
        """Create a RecordSet."""
        return RecordSet(
            parameters={
                self.get_str(): self.parameters_record()
                for _ in range(num_params_records)
            },
            metrics={
                self.get_str(): self.metrics_record()
                for _ in range(num_metrics_records)
            },
            configs={
                self.get_str(): self.configs_record()
                for _ in range(num_configs_records)
            },
        )

    def metadata(self) -> Metadata:
        """Create a Metadata."""
        return Metadata(
            run_id=self.rng.randint(0, 1 << 30),
            task_id=self.get_str(64),
            group_id=self.get_str(30),
            ttl=self.get_str(10),
            task_type=self.get_str(10),
        )


def test_array_serialization_deserialization() -> None:
    """Test serialization and deserialization of Array."""
    # Prepare
    maker = RecordMaker()
    original = maker.array()

    # Execute
    proto = array_to_proto(original)
    deserialized = array_from_proto(proto)

    # Assert
    assert isinstance(proto, ProtoArray)
    assert original == deserialized


def test_parameters_record_serialization_deserialization() -> None:
    """Test serialization and deserialization of ParametersRecord."""
    # Prepare
    maker = RecordMaker()
    original = maker.parameters_record()

    # Execute
    proto = parameters_record_to_proto(original)
    deserialized = parameters_record_from_proto(proto)

    # Assert
    assert isinstance(proto, ProtoParametersRecord)
    assert original.data == deserialized.data


def test_metrics_record_serialization_deserialization() -> None:
    """Test serialization and deserialization of MetricsRecord."""
    # Prepare
    maker = RecordMaker()
    original = maker.metrics_record()

    # Execute
    proto = metrics_record_to_proto(original)
    deserialized = metrics_record_from_proto(proto)

    # Assert
    assert isinstance(proto, ProtoMetricsRecord)
    assert original.data == deserialized.data


def test_configs_record_serialization_deserialization() -> None:
    """Test serialization and deserialization of ConfigsRecord."""
    # Prepare
    maker = RecordMaker()
    original = maker.configs_record()

    # Execute
    proto = configs_record_to_proto(original)
    deserialized = configs_record_from_proto(proto)

    # Assert
    assert isinstance(proto, ProtoConfigsRecord)
    assert original.data == deserialized.data


def test_recordset_serialization_deserialization() -> None:
    """Test serialization and deserialization of RecordSet."""
    # Prepare
    maker = RecordMaker(state=0)
    original = maker.recordset(2, 2, 1)

    # Execute
    proto = recordset_to_proto(original)
    deserialized = recordset_from_proto(proto)

    # Assert
    assert isinstance(proto, ProtoRecordSet)
    assert original == deserialized


def test_message_to_and_from_taskins() -> None:
    """Test Message to and from TaskIns."""
    # Prepare
    maker = RecordMaker(state=1)
    metadata = maker.metadata()
    original = Message(
        metadata=Metadata(
            run_id=0,
            task_id="",
            group_id="",
            ttl=metadata.ttl,
            task_type=metadata.task_type,
        ),
        message=maker.recordset(1, 1, 1),
    )

    # Execute
    taskins = message_to_taskins(original)
    taskins.run_id = metadata.run_id
    taskins.task_id = metadata.task_id
    taskins.group_id = metadata.group_id
    deserialized = message_from_taskins(taskins)

    # Assert
    assert original.message == deserialized.message
    assert metadata == deserialized.metadata


def test_message_to_and_from_taskres() -> None:
    """Test Message to and from TaskRes."""
    # Prepare
    maker = RecordMaker(state=2)
    metadata = maker.metadata()
    original = Message(
        metadata=Metadata(
            run_id=0,
            task_id="",
            group_id="",
            ttl=metadata.ttl,
            task_type=metadata.task_type,
        ),
        message=maker.recordset(1, 1, 1),
    )

    # Execute
    taskres = message_to_taskres(original)
    taskres.run_id = metadata.run_id
    taskres.task_id = metadata.task_id
    taskres.group_id = metadata.group_id
    deserialized = message_from_taskres(taskres)

    # Assert
    assert original.message == deserialized.message
    assert metadata == deserialized.metadata
