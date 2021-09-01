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
"""Parameter conversion."""


from io import BytesIO
from typing import cast

import numpy as np

from .typing import Parameters, Weights


def weights_to_parameters(weights: Weights) -> Parameters:
    """Convert NumPy weights to parameters object."""
    tensors = [ndarray_to_bytes(ndarray) for ndarray in weights]
    return Parameters(tensors=tensors, tensor_type="numpy.ndarray")


def parameters_to_weights(parameters: Parameters) -> Weights:
    """Convert parameters object to NumPy weights."""
    return [
        bytes_to_ndarray(tensor, parameters.tensor_type)
        for tensor in parameters.tensors
    ]


def ndarray_to_bytes(ndarray: np.ndarray) -> bytes:
    """Serialize NumPy ndarray to bytes."""
    bytes_io = BytesIO()
    np.save(bytes_io, ndarray, allow_pickle=False)
    return bytes_io.getvalue()


def bytes_to_ndarray(tensor: bytes, tensor_type: str = "numpy.ndarray") -> np.ndarray:
    """Deserialize bytes into numpy arrays according to incoming `tensor_type`.

    Args:
        tensor (bytes): Bytes object encoding weights.
        tensor_type (str, optional): String describing the original format of bytes.
            Defaults to "numpy.ndarray".

    Returns:
       weights (np.ndarray): Weights in numpy array format.
    """
    base_types = {"float32": np.float32, "float64": np.float64}
    weights = None
    if tensor_type == "numpy.ndarray":
        bytes_io = BytesIO(tensor)
        ndarray_deserialized = np.load(bytes_io, allow_pickle=False)
        weights = cast(np.ndarray, ndarray_deserialized)
    else:
        try:
            dtype = base_types[tensor_type]
            weights = np.frombuffer(tensor, dtype=dtype.newbyteorder(">"))
        except KeyError:
            print(f"tensor_type '{tensor_type}' unknown.")
    return weights
