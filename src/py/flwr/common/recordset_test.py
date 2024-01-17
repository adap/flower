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


import numpy as np

from .parameter import ndarrays_to_parameters, parameters_to_ndarrays
from .parametersrecord import Array
from .recordset_utils import (
    parameters_to_parametersrecord,
    parametersrecord_to_parameters,
)
from .typing import NDArray, NDArrays, Parameters


def get_ndarrays() -> NDArrays:
    """Return list of NumPy arrays."""
    arr1 = np.array([[1.0, 2.0], [3.0, 4], [5.0, 6.0]])
    arr2 = np.eye(2, 7, 3)

    return [arr1, arr2]


def ndarray_to_array(ndarray: NDArray) -> Array:
    """Represent NumPy ndarray as Array."""
    return Array(
        data=ndarray.tobytes(),
        dtype=str(ndarray.dtype),
        stype="numpy.ndarray.tobytes",
        shape=list(ndarray.shape),
    )


def test_ndarray_to_array() -> None:
    """Test creation of Array object from NumPy ndarray."""
    shape = (2, 7, 9)
    arr = np.eye(*shape)

    array = ndarray_to_array(arr)

    arr_ = np.frombuffer(buffer=array.data, dtype=array.dtype).reshape(array.shape)

    assert np.array_equal(arr, arr_)


def test_parameters_to_array_and_back() -> None:
    """Test conversion between legacy Parameters and Array."""
    ndarrays = get_ndarrays()

    # Array represents a single array, unlike Paramters, which represent a
    # list of arrays
    ndarray = ndarrays[0]

    parameters = ndarrays_to_parameters([ndarray])

    array = Array(
        data=parameters.tensors[0], dtype="", stype=parameters.tensor_type, shape=[]
    )

    parameters = Parameters(tensors=[array.data], tensor_type=array.stype)

    ndarray_ = parameters_to_ndarrays(parameters=parameters)[0]

    assert np.array_equal(ndarray, ndarray_)


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
        assert np.array_equal(arr, arr_)


# def test_torch_statedict_to_parametersrecord() -> None:
#     """."""
#     import torch
#     from .parametersrecord import ParametersRecord

#     layer = torch.nn.Conv2d(3, 5, 16)
#     layer_sd = layer.state_dict()

#     p_c = ParametersRecord()

#     for k in layer_sd.keys():
#         layer_sd[k] = ndarray_to_array(layer_sd[k].numpy())

#     p_c.add_parameters(layer_sd)
