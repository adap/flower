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

import pickle
from collections import namedtuple
from copy import deepcopy
from typing import Callable, Dict, List, OrderedDict, Type, Union

import numpy as np
import pytest

from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.recordset_compat import (
    parameters_to_parametersrecord,
    parametersrecord_to_parameters,
)
from flwr.common.typing import (
    ConfigsRecordValues,
    MetricsRecordValues,
    NDArray,
    NDArrays,
    Parameters,
)

from . import Array, ConfigsRecord, MetricsRecord, ParametersRecord, RecordSet


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

    # Array represents a single array, unlike parameters, which represent a
    # list of arrays
    ndarray = ndarrays[0]

    parameters = ndarrays_to_parameters([ndarray])

    array = Array(
        data=parameters.tensors[0], dtype="", stype=parameters.tensor_type, shape=[]
    )

    parameters = Parameters(tensors=[array.data], tensor_type=array.stype)

    ndarray_ = parameters_to_ndarrays(parameters=parameters)[0]

    assert np.array_equal(ndarray, ndarray_)


@pytest.mark.parametrize(
    "keep_input, validate_freed_fn",
    [
        (False, lambda x, x_copy, y: len(x.tensors) == 0),  # check tensors were freed
        (True, lambda x, x_copy, y: x.tensors == y.tensors),  # check they are equal
    ],
)
def test_parameters_to_parametersrecord_and_back(
    keep_input: bool,
    validate_freed_fn: Callable[[Parameters, Parameters, Parameters], bool],
) -> None:
    """Test conversion between legacy Parameters and ParametersRecords."""
    ndarrays = get_ndarrays()

    parameters = ndarrays_to_parameters(ndarrays)
    parameters_copy = deepcopy(parameters)

    params_record = parameters_to_parametersrecord(
        parameters=parameters, keep_input=keep_input
    )

    parameters_ = parametersrecord_to_parameters(params_record, keep_input=keep_input)

    ndarrays_ = parameters_to_ndarrays(parameters=parameters_)

    # Validate returned NDArrays match those at the beginning
    for arr, arr_ in zip(ndarrays, ndarrays_):
        assert np.array_equal(arr, arr_), "no"

    # Validate initial Parameters object has been handled according to `keep_input`
    assert validate_freed_fn(parameters, parameters_copy, parameters_)


def test_set_parameters_while_keeping_intputs() -> None:
    """Tests keep_input functionality in ParametersRecord."""
    # Adding parameters to a record that doesn't erase entries in the input `array_dict`
    array_dict = OrderedDict(
        {str(i): ndarray_to_array(ndarray) for i, ndarray in enumerate(get_ndarrays())}
    )
    p_record = ParametersRecord(array_dict, keep_input=True)

    # Creating a second parametersrecord passing the same `array_dict` (not erased)
    p_record_2 = ParametersRecord(array_dict)
    assert p_record == p_record_2

    # Now it should be empty (the second ParametersRecord wasn't flagged to keep it)
    assert len(array_dict) == 0


def test_set_parameters_with_correct_types() -> None:
    """Test adding dictionary of Arrays to ParametersRecord."""
    p_record = ParametersRecord()
    array_dict = OrderedDict(
        {str(i): ndarray_to_array(ndarray) for i, ndarray in enumerate(get_ndarrays())}
    )
    p_record.update(array_dict)


@pytest.mark.parametrize(
    "key_type, value_fn",
    [
        (str, lambda x: x),  # correct key, incorrect value
        (str, lambda x: x.tolist()),  # correct key, incorrect value
        (int, ndarray_to_array),  # incorrect key, correct value
        (int, lambda x: x),  # incorrect key, incorrect value
        (int, lambda x: x.tolist()),  # incorrect key, incorrect value
    ],
)
def test_set_parameters_with_incorrect_types(
    key_type: Type[Union[int, str]],
    value_fn: Callable[[NDArray], Union[NDArray, List[float]]],
) -> None:
    """Test adding dictionary of unsupported types to ParametersRecord."""
    p_record = ParametersRecord()

    array_dict = {
        key_type(i): value_fn(ndarray) for i, ndarray in enumerate(get_ndarrays())
    }

    with pytest.raises(TypeError):
        p_record.update(array_dict)


@pytest.mark.parametrize(
    "key_type, value_fn",
    [
        (str, lambda x: int(x.flatten()[0])),  # str: int
        (str, lambda x: float(x.flatten()[0])),  # str: float
        (str, lambda x: x.flatten().astype("int").tolist()),  # str: List[int]
        (str, lambda x: x.flatten().astype("float").tolist()),  # str: List[float]
        (str, lambda x: []),  # str: empty list
    ],
)
def test_set_metrics_to_metricsrecord_with_correct_types(
    key_type: Type[str],
    value_fn: Callable[[NDArray], MetricsRecordValues],
) -> None:
    """Test adding metrics of various types to a MetricsRecord."""
    m_record = MetricsRecord()

    labels = [1, 2.0]
    arrays = get_ndarrays()

    my_metrics = OrderedDict(
        {key_type(label): value_fn(arr) for label, arr in zip(labels, arrays)}
    )

    # Add metric
    m_record.update(my_metrics)

    # Check metrics are actually added
    assert my_metrics == m_record


@pytest.mark.parametrize(
    "key_type, value_fn",
    [
        (str, lambda x: str(x.flatten()[0])),  # str: str  (supported: unsupported)
        (str, lambda x: bool(x.flatten()[0])),  # str: bool  (supported: unsupported)
        (
            str,
            lambda x: x.flatten().astype("str").tolist(),
        ),  # str: List[str] (supported: unsupported)
        (str, lambda x: x),  # str: NDArray (supported: unsupported)
        (
            str,
            lambda x: {str(v): v for v in x.flatten()},
        ),  # str: dict[str: float] (supported: unsupported)
        (
            str,
            lambda x: [{str(v): v for v in x.flatten()}],
        ),  # str: List[dict[str: float]] (supported: unsupported)
        (
            str,
            lambda x: [1, 2.0, 3.0, 4],
        ),  # str: List[mixing valid types] (supported: unsupported)
        (
            int,
            lambda x: x.flatten().tolist(),
        ),  # int: List[str] (unsupported: supported)
        (
            float,
            lambda x: x.flatten().tolist(),
        ),  # float: List[int] (unsupported: supported)
    ],
)
def test_set_metrics_to_metricsrecord_with_incorrect_types(
    key_type: Type[Union[str, int, float, bool]],
    value_fn: Callable[[NDArray], Union[NDArray, Dict[str, NDArray], List[float]]],
) -> None:
    """Test adding metrics of various unsupported types to a MetricsRecord."""
    m_record = MetricsRecord()

    labels = [1, 2.0]
    arrays = get_ndarrays()

    my_metrics = OrderedDict(
        {key_type(label): value_fn(arr) for label, arr in zip(labels, arrays)}
    )

    with pytest.raises(TypeError):
        m_record.update(my_metrics)


@pytest.mark.parametrize(
    "keep_input",
    [
        (True),
        (False),
    ],
)
def test_set_metrics_to_metricsrecord_with_and_without_keeping_input(
    keep_input: bool,
) -> None:
    """Test keep_input functionality for MetricsRecord."""
    # constructing a valid input
    labels = [1, 2.0]
    arrays = get_ndarrays()
    my_metrics = OrderedDict(
        {str(label): arr.flatten().tolist() for label, arr in zip(labels, arrays)}
    )

    my_metrics_copy = my_metrics.copy()

    # Add metric
    m_record = MetricsRecord(my_metrics, keep_input=keep_input)

    # Check metrics are actually added
    # Check that input dict has been emptied when enabled such behaviour
    if keep_input:
        assert my_metrics == m_record
    else:
        assert my_metrics_copy == m_record
        assert len(my_metrics) == 0


@pytest.mark.parametrize(
    "key_type, value_fn",
    [
        (str, lambda x: str(x.flatten()[0])),  # str: str
        (str, lambda x: int(x.flatten()[0])),  # str: int
        (str, lambda x: float(x.flatten()[0])),  # str: float
        (str, lambda x: bool(x.flatten()[0])),  # str: bool
        (str, lambda x: x.flatten().tobytes()),  # str: bytes
        (str, lambda x: x.flatten().astype("str").tolist()),  # str: List[str]
        (str, lambda x: x.flatten().astype("int").tolist()),  # str: List[int]
        (str, lambda x: x.flatten().astype("float").tolist()),  # str: List[float]
        (str, lambda x: x.flatten().astype("bool").tolist()),  # str: List[bool]
        (str, lambda x: [x.flatten().tobytes()]),  # str: List[bytes]
        (str, lambda x: []),  # str: emptyt list
    ],
)
def test_set_configs_to_configsrecord_with_correct_types(
    key_type: Type[str],
    value_fn: Callable[[NDArray], ConfigsRecordValues],
) -> None:
    """Test adding configs of various types to a ConfigsRecord."""
    labels = [1, 2.0]
    arrays = get_ndarrays()

    my_configs = OrderedDict(
        {key_type(label): value_fn(arr) for label, arr in zip(labels, arrays)}
    )

    c_record = ConfigsRecord(my_configs)

    # check values are actually there
    assert c_record == my_configs


@pytest.mark.parametrize(
    "key_type, value_fn",
    [
        (str, lambda x: x),  # str: NDArray (supported: unsupported)
        (
            str,
            lambda x: {str(v): v for v in x.flatten()},
        ),  # str: dict[str: float] (supported: unsupported)
        (
            str,
            lambda x: [{str(v): v for v in x.flatten()}],
        ),  # str: List[dict[str: float]] (supported: unsupported)
        (
            str,
            lambda x: [1, 2.0, 3.0, 4],
        ),  # str: List[mixing valid types] (supported: unsupported)
        (
            int,
            lambda x: x.flatten().tolist(),
        ),  # int: List[str] (unsupported: supported)
        (
            float,
            lambda x: x.flatten().tolist(),
        ),  # float: List[int] (unsupported: supported)
    ],
)
def test_set_configs_to_configsrecord_with_incorrect_types(
    key_type: Type[Union[str, int, float]],
    value_fn: Callable[[NDArray], Union[NDArray, Dict[str, NDArray], List[float]]],
) -> None:
    """Test adding configs of various unsupported types to a ConfigsRecord."""
    c_record = ConfigsRecord()

    labels = [1, 2.0]
    arrays = get_ndarrays()

    my_configs = OrderedDict(
        {key_type(label): value_fn(arr) for label, arr in zip(labels, arrays)}
    )

    with pytest.raises(TypeError):
        c_record.update(my_configs)


def test_count_bytes_metricsrecord() -> None:
    """Test counting bytes in MetricsRecord."""
    data = {"a": 1, "b": 2.0, "c": [1, 2, 3], "d": [1.0, 2.0, 3.0, 4.0, 5.0]}
    bytes_in_dict = 8 + 8 + 3 * 8 + 5 * 8
    bytes_in_dict += 4  # represnting the keys

    m_record = MetricsRecord()
    m_record.update(OrderedDict(data))
    record_bytest_count = m_record.count_bytes()
    assert bytes_in_dict == record_bytest_count


def test_count_bytes_configsrecord() -> None:
    """Test counting bytes in ConfigsRecord."""
    data = {"a": 1, "b": 2.0, "c": [1, 2, 3], "d": [1.0, 2.0, 3.0, 4.0, 5.0]}
    bytes_in_dict = 8 + 8 + 3 * 8 + 5 * 8
    bytes_in_dict += 4  # represnting the keys

    to_add = {
        "aa": True,
        "bb": "False",
        "cc": bytes(9),
        "dd": [True, False, False],
        "ee": ["True", "False"],
        "ff": [bytes(1), bytes(13), bytes(51)],
    }
    data = {**data, **to_add}
    bytes_in_dict += 1 + 5 + 9 + 3 + (4 + 5) + (1 + 13 + 51)
    bytes_in_dict += 12  # represnting the keys

    bytes_in_dict = int(bytes_in_dict)

    c_record = ConfigsRecord()
    c_record.update(OrderedDict(data))

    record_bytest_count = c_record.count_bytes()
    assert bytes_in_dict == record_bytest_count


def test_record_is_picklable() -> None:
    """Test if RecordSet and *Record are picklable."""
    # Prepare
    p_record = ParametersRecord()
    m_record = MetricsRecord({"aa": 123})
    c_record = ConfigsRecord({"cc": bytes(9)})
    rs = RecordSet()
    rs.parameters_records["params"] = p_record
    rs.metrics_records["metrics"] = m_record
    rs.configs_records["configs"] = c_record

    # Execute
    pickle.dumps((p_record, m_record, c_record, rs))


def test_recordset_repr() -> None:
    """Test the string representation of RecordSet."""
    # Prepare
    kwargs = {
        "parameters_records": {"params": ParametersRecord()},
        "metrics_records": {"metrics": MetricsRecord({"aa": 123})},
        "configs_records": {"configs": ConfigsRecord({"cc": bytes(9)})},
    }
    rs = RecordSet(**kwargs)  # type: ignore
    expected = namedtuple("RecordSet", kwargs.keys())(**kwargs)

    # Assert
    assert str(rs) == str(expected)