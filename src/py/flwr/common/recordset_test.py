# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""RecordSet tests."""
import secrets

import numpy as np

from .parameter import ndarrays_to_parameters, parameters_to_ndarrays
from .parametersrecord import Array
from .recordset_utils import (
    parameters_to_parametersrecord,
    parametersrecord_to_parameters,
)
from .typing import NDArrays, Parameters


def get_ndarrays() -> NDArrays:
    """Return list of NumPy arrays."""
    arr1 = np.array([[1.0, 2.0], [3.0, 4], [5.0, 6.0]])
    arr2 = np.eye(2, 7, 3)

    return [arr1, arr2]


def test_ndarray_to_array() -> None:
    """Test creation of Array object from NumPy array."""
    shape = (2, 7, 9)
    arr = np.eye(*shape)

    tensor = Array(
        arr.tobytes(),
        dtype=str(arr.dtype),
        stype="np.tobytes",
        shape=list(arr.shape),
        ref=secrets.token_hex(16),
    )

    arr_ = np.frombuffer(buffer=tensor.data, dtype=tensor.dtype).reshape(tensor.shape)

    assert np.allclose(arr, arr_)


def test_parameters_to_array_and_back() -> None:
    """Test conversion between legacy Parameters and Array."""
    ndarrays = get_ndarrays()

    # Array represents a single array, unlike Paramters, which represent a
    # list of arrays
    ndarray = ndarrays[0]

    parameters = ndarrays_to_parameters([ndarray])

    array = Array(
        data=parameters.tensors[0], dtype=parameters.tensor_type, stype="", shape=[]
    )

    parameters = Parameters(tensors=[array.data], tensor_type=array.dtype)

    ndarray_ = parameters_to_ndarrays(parameters=parameters)[0]

    assert np.allclose(ndarray, ndarray_)


def test_parameters_to_parametersrecord_and_back() -> None:
    """Test utility function to convert between legacy Parameters.

    and ParametersRecords.
    """
    ndarrays = get_ndarrays()

    parameters = ndarrays_to_parameters(ndarrays)

    params_record = parameters_to_parametersrecord(parameters=parameters)

    parameters_ = parametersrecord_to_parameters(params_record)

    ndarrays_ = parameters_to_ndarrays(parameters=parameters_)

    for arr, arr_ in zip(ndarrays, ndarrays_):
        assert np.allclose(arr, arr_)
