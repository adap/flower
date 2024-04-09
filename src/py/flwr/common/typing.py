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
"""Flower type definitions."""


from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
import uuid

import numpy as np
import numpy.typing as npt

from flwr.common.aws import BucketManager
import json

import zlib

NDArray = npt.NDArray[Any]
NDArrayInt = npt.NDArray[np.int_]
NDArrayFloat = npt.NDArray[np.float_]
NDArrays = List[NDArray]

# The following union type contains Python types corresponding to ProtoBuf types that
# ProtoBuf considers to be "Scalar Value Types", even though some of them arguably do
# not conform to other definitions of what a scalar is. Source:
# https://developers.google.com/protocol-buffers/docs/overview#scalar
Scalar = Union[bool, bytes, float, int, str]
Value = Union[
    bool,
    bytes,
    float,
    int,
    str,
    List[bool],
    List[bytes],
    List[float],
    List[int],
    List[str],
]

# Value types for common.MetricsRecord
MetricsScalar = Union[int, float]
MetricsScalarList = Union[List[int], List[float]]
MetricsRecordValues = Union[MetricsScalar, MetricsScalarList]
# Value types for common.ConfigsRecord
ConfigsScalar = Union[MetricsScalar, str, bytes, bool]
ConfigsScalarList = Union[MetricsScalarList, List[str], List[bytes], List[bool]]
ConfigsRecordValues = Union[ConfigsScalar, ConfigsScalarList]

Metrics = Dict[str, Scalar]
MetricsAggregationFn = Callable[[List[Tuple[int, Metrics]]], Metrics]

Config = Dict[str, Scalar]
Properties = Dict[str, Scalar]


class Code(Enum):
    """Client status codes."""

    OK = 0
    GET_PROPERTIES_NOT_IMPLEMENTED = 1
    GET_PARAMETERS_NOT_IMPLEMENTED = 2
    FIT_NOT_IMPLEMENTED = 3
    EVALUATE_NOT_IMPLEMENTED = 4


@dataclass
class Status:
    """Client status."""

    code: Code
    message: str


@dataclass
class Parameters:
    """Model parameters."""

    tensors: List[bytes]
    tensor_type: str
    s3_object_key: Optional[uuid.UUID] = None


    @property
    def dimensions(self) -> List[int]:
        return [len(x) for x in self.tensors]
    
    def compressed_tensor_bytes(self):
        return zlib.compress(b''.join(self.tensors))

    def upload_to_s3(self, bucket_manager: BucketManager):
        # Parameter can only be uploaded once. This saves bandwidth if user attempts try to upload the same parameter object more than oncce

        if self.s3_object_key is not None:
            return

        s3_object_key = uuid.uuid4()
        body = self.compressed_tensor_bytes()
        bucket_manager.bucket.put_object(
            Key=str(s3_object_key),
            Metadata=dict(
                tensor_type=self.tensor_type,
                dimensions=json.dumps(self.dimensions)
            ),
            Body=body
        ).wait_until_exists()
        self.s3_object_key = s3_object_key
    
    @staticmethod
    def pull_from_s3(bucket_manager: BucketManager, id: uuid.UUID):
        key = str(id)
        result = bucket_manager.bucket.Object(key).get()
        tensor_type = result["Metadata"]["tensor_type"]
        dimensions = json.loads(result["Metadata"]["dimensions"])
        compressed_tensors_bytes = result["Body"].read()

        params = Parameters.from_bytes(
            tensor_type,
            compressed_tensors_bytes,
            dimensions
        )
        params.s3_object_key = id

        return params

    @staticmethod
    def from_bytes(
        tensor_type: str,
        compressed_tensors_bytes: bytes,
        dimensions: List[int]
    ):
        tensors_bytes = zlib.decompress(compressed_tensors_bytes)
        assert len(tensors_bytes) == sum(dimensions)

        last_ptr = 0
        tensors = []

        for dim in dimensions:
            tensors.append(tensors_bytes[last_ptr: last_ptr+dim])
            last_ptr += dim
        
        return Parameters(
            tensors=tensors,
            tensor_type=tensor_type
        )



@dataclass
class GetParametersIns:
    """Parameters request for a client."""

    config: Config


@dataclass
class GetParametersRes:
    """Response when asked to return parameters."""

    status: Status
    parameters: Parameters


@dataclass
class FitIns:
    """Fit instructions for a client."""

    parameters: Parameters
    config: Dict[str, Scalar]


@dataclass
class FitRes:
    """Fit response from a client."""

    status: Status
    parameters: Parameters
    num_examples: int
    metrics: Optional[Dict[str, Scalar]]


@dataclass
class EvaluateIns:
    """Evaluate instructions for a client."""

    parameters: Parameters
    config: Dict[str, Scalar]


@dataclass
class EvaluateRes:
    """Evaluate response from a client."""

    status: Status
    loss: float
    num_examples: int
    metrics: Optional[Dict[str, Scalar]]


@dataclass
class GetPropertiesIns:
    """Properties request for a client."""

    config: Config


@dataclass
class GetPropertiesRes:
    """Properties response from a client."""

    status: Status
    properties: Properties


@dataclass
class ReconnectIns:
    """ReconnectIns message from server to client."""

    seconds: Optional[int]


@dataclass
class DisconnectRes:
    """DisconnectRes message from client to server."""

    reason: str


@dataclass
class ServerMessage:
    """ServerMessage is a container used to hold one instruction message."""

    get_properties_ins: Optional[GetPropertiesIns] = None
    get_parameters_ins: Optional[GetParametersIns] = None
    fit_ins: Optional[FitIns] = None
    evaluate_ins: Optional[EvaluateIns] = None


@dataclass
class ClientMessage:
    """ClientMessage is a container used to hold one result message."""

    get_properties_res: Optional[GetPropertiesRes] = None
    get_parameters_res: Optional[GetParametersRes] = None
    fit_res: Optional[FitRes] = None
    evaluate_res: Optional[EvaluateRes] = None
