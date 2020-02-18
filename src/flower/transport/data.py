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
"""Helper functions for protobuf serialization and deserialization"""

from io import BytesIO

import numpy as np

from flower.proto.transport_pb2 import NDArray


def ndarray_to_proto(ndarr: np.ndarray) -> NDArray:
    """Serializes a numpy array to NDArray protobuf message"""
    ndarr_bytes = BytesIO()
    np.save(ndarr_bytes, ndarr, allow_pickle=False)
    return NDArray(ndarray=ndarr_bytes.getvalue())


def proto_to_ndarray(ndarr_proto: NDArray) -> np.ndarray:
    """Deserializes a NDArray protobuf message to a numpy array"""
    ndarr_bytes = BytesIO(ndarr_proto.ndarray)
    return np.load(ndarr_bytes, allow_pickle=False)
