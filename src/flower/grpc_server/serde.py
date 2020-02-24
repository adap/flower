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

from io import BytesIO
from typing import cast

import numpy as np

from flower.proto.transport_pb2 import NDArray


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
