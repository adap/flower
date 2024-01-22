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

from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, List, OrderedDict, Type, Union

import numpy as np
import pytest

from .configsrecord import ConfigsRecord
from .flowercontext import FlowerContext, Metadata
from .metricsrecord import MetricsRecord
from .parameter import ndarrays_to_parameters, parameters_to_ndarrays
from .parametersrecord import Array, ParametersRecord
from .recordset import RecordSet
from .recordset_utils import (
    evaluate_ins_to_recordset,
    evaluate_res_to_recordset,
    fit_ins_to_recordset,
    fit_res_to_recordset,
    getparameters_ins_to_recordset,
    getparameters_res_to_recordset,
    getproperties_ins_to_recordset,
    getproperties_res_to_recordset,
    parameters_to_parametersrecord,
    parametersrecord_to_parameters,
    recordset_to_evaluate_ins,
    recordset_to_evaluate_res,
    recordset_to_fit_ins,
    recordset_to_fit_res,
    recordset_to_getparameters_ins,
    recordset_to_getparameters_res,
    recordset_to_getproperties_ins,
    recordset_to_getproperties_res,
)
from .typing import (
    Code,
    ConfigsRecordValues,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    MetricsRecordValues,
    NDArray,
    NDArrays,
    Parameters,
    Scalar,
    Status,
)


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
    """Test conversion between legacy Parameters and ParametersRecords."""
    ndarrays = get_ndarrays()

    parameters = ndarrays_to_parameters(ndarrays)

    params_record = parameters_to_parametersrecord(parameters=parameters)

    parameters_ = parametersrecord_to_parameters(params_record)

    ndarrays_ = parameters_to_ndarrays(parameters=parameters_)

    for arr, arr_ in zip(ndarrays, ndarrays_):
        assert np.array_equal(arr, arr_)


def test_set_parameters_while_keeping_intputs() -> None:
    """Tests keep_input functionality in ParametersRecord."""
    # Adding parameters to a record that doesn't erase entries in the input `array_dict`
    p_record = ParametersRecord(keep_input=True)
    array_dict = OrderedDict(
        {str(i): ndarray_to_array(ndarray) for i, ndarray in enumerate(get_ndarrays())}
    )
    p_record.set_parameters(array_dict, keep_input=True)

    # Creating a second parametersrecord passing the same `array_dict` (not erased)
    p_record_2 = ParametersRecord(array_dict)
    assert p_record.data == p_record_2.data

    # Now it should be empty (the second ParametersRecord wasn't flagged to keep it)
    assert len(array_dict) == 0


def test_set_parameters_with_correct_types() -> None:
    """Test adding dictionary of Arrays to ParametersRecord."""
    p_record = ParametersRecord()
    array_dict = OrderedDict(
        {str(i): ndarray_to_array(ndarray) for i, ndarray in enumerate(get_ndarrays())}
    )
    p_record.set_parameters(array_dict)


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
        p_record.set_parameters(array_dict)  # type: ignore


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
    m_record.set_metrics(my_metrics)

    # Check metrics are actually added
    assert my_metrics == m_record.data


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
        m_record.set_metrics(my_metrics)  # type: ignore


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
    m_record = MetricsRecord(keep_input=keep_input)

    # constructing a valid input
    labels = [1, 2.0]
    arrays = get_ndarrays()
    my_metrics = OrderedDict(
        {str(label): arr.flatten().tolist() for label, arr in zip(labels, arrays)}
    )

    my_metrics_copy = my_metrics.copy()

    # Add metric
    m_record.set_metrics(my_metrics, keep_input=keep_input)

    # Check metrics are actually added
    # Check that input dict has been emptied when enabled such behaviour
    if keep_input:
        assert my_metrics == m_record.data
    else:
        assert my_metrics_copy == m_record.data
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
        (str, lambda x: []),  # str: empyt list
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
    assert c_record.data == my_configs


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
    m_record = ConfigsRecord()

    labels = [1, 2.0]
    arrays = get_ndarrays()

    my_metrics = OrderedDict(
        {key_type(label): value_fn(arr) for label, arr in zip(labels, arrays)}
    )

    with pytest.raises(TypeError):
        m_record.set_configs(my_metrics)  # type: ignore


##################################################
#  Testing conversion: *Ins --> RecordSet --> *Ins
#  Testing conversion: *Res <-- RecordSet <-- *Res
##################################################


def _get_valid_fitins() -> FitIns:
    arrays = get_ndarrays()
    return FitIns(parameters=ndarrays_to_parameters(arrays), config={"a": 1.0, "b": 0})


def _get_valid_evaluateins() -> EvaluateIns:
    fit_ins = _get_valid_fitins()
    return EvaluateIns(parameters=fit_ins.parameters, config=fit_ins.config)


def _get_valid_getparametersins() -> GetParametersIns:
    config_dict: Dict[str, Scalar] = {
        "a": 1.0,
        "b": 3,
        "c": True,
    }  # valid since both Ins/Res communicate over ConfigsRecord

    return GetParametersIns(config_dict)


def _get_valid_getpropertiesins() -> GetPropertiesIns:
    getparamsins = _get_valid_getparametersins()
    return GetPropertiesIns(config=getparamsins.config)


def test_fitins_to_recordset_and_back() -> None:
    """Test conversion FitIns --> RecordSet --> FitIns."""
    fitins = _get_valid_fitins()

    fitins_copy = deepcopy(fitins)

    recordset = fit_ins_to_recordset(fitins, keep_input=False)

    fitins_ = recordset_to_fit_ins(recordset, keep_input=False)

    assert fitins_copy == fitins_


@pytest.mark.parametrize(
    "context, metrics",
    [
        (nullcontext(), {"a": 1.0, "b": 0}),
        (
            pytest.raises(TypeError),
            {"a": 1.0, "b": 3, "c": True},
        ),  # fails due to unsupported type for metricsrecord value
    ],
)
def test_fitres_to_recordset_and_back(context: Any, metrics: Dict[str, Scalar]) -> None:
    """Test conversion FitRes --> RecordSet --> FitRes."""
    arrays = get_ndarrays()
    fitres = FitRes(
        parameters=ndarrays_to_parameters(arrays),
        num_examples=1,
        status=Status(code=Code(0), message=""),
        metrics=metrics,
    )

    fitres_copy = deepcopy(fitres)

    with context:
        recordset = fit_res_to_recordset(fitres, keep_input=False)
        fitres_ = recordset_to_fit_res(recordset, keep_input=False)

    # only check if we didn't test for an invalid setting. Only in valid settings
    # makes sense to evaluate the below, since both functions above have succesfully
    # being executed.
    if isinstance(context, nullcontext):
        assert fitres_copy == fitres_


def test_evaluateins_to_recordset_and_back() -> None:
    """Test conversion EvaluateIns --> RecordSet --> EvaluateIns."""
    evaluateins = _get_valid_evaluateins()

    evaluateins_copy = deepcopy(evaluateins)

    recordset = evaluate_ins_to_recordset(evaluateins, keep_input=False)

    evaluateins_ = recordset_to_evaluate_ins(recordset, keep_input=False)

    assert evaluateins_copy == evaluateins_


@pytest.mark.parametrize(
    "context, metrics",
    [
        (nullcontext(), {"a": 1.0, "b": 0}),
        (
            pytest.raises(TypeError),
            {"a": 1.0, "b": 3, "c": True},
        ),  # fails due to unsupported type for metricsrecord value
    ],
)
def test_evaluateres_to_recordset_and_back(
    context: Any, metrics: Dict[str, Scalar]
) -> None:
    """Test conversion EvaluateRes --> RecordSet --> EvaluateRes."""
    evaluateres = EvaluateRes(
        num_examples=1,
        loss=0.1,
        status=Status(code=Code(0), message=""),
        metrics=metrics,
    )

    evaluateres_copy = deepcopy(evaluateres)

    with context:
        recordset = evaluate_res_to_recordset(evaluateres)
        evaluateres_ = recordset_to_evaluate_res(recordset)

    # only check if we didn't test for an invalid setting. Only in valid settings
    # makes sense to evaluate the below, since both functions above have succesfully
    # being executed.
    if isinstance(context, nullcontext):
        assert evaluateres_copy == evaluateres_


def test_get_properties_ins_to_recordset_and_back() -> None:
    """Test conversion GetPropertiesIns --> RecordSet --> GetPropertiesIns."""
    getproperties_ins = _get_valid_getpropertiesins()

    getproperties_ins_copy = deepcopy(getproperties_ins)

    recordset = getproperties_ins_to_recordset(getproperties_ins)
    getproperties_ins_ = recordset_to_getproperties_ins(recordset)

    assert getproperties_ins_copy == getproperties_ins_


def test_get_properties_res_to_recordset_and_back() -> None:
    """Test conversion GetPropertiesRes --> RecordSet --> GetPropertiesRes."""
    config_dict: Dict[str, Scalar] = {
        "a": 1.0,
        "b": 3,
        "c": True,
    }  # valid since both Ins/Res communicate over ConfigsRecord

    getproperties_res = GetPropertiesRes(
        status=Status(code=Code(0), message=""), properties=config_dict
    )

    getproperties_res_copy = deepcopy(getproperties_res)

    recordset = getproperties_res_to_recordset(getproperties_res)
    getproperties_res_ = recordset_to_getproperties_res(recordset)

    assert getproperties_res_copy == getproperties_res_


def test_get_parameters_ins_to_recordset_and_back() -> None:
    """Test conversion GetParametersIns --> RecordSet --> GetParametersIns."""
    getparameters_ins = _get_valid_getparametersins()

    getparameters_ins_copy = deepcopy(getparameters_ins)

    recordset = getparameters_ins_to_recordset(getparameters_ins)
    getparameters_ins_ = recordset_to_getparameters_ins(recordset)

    assert getparameters_ins_copy == getparameters_ins_


def test_get_parameters_res_to_recordset_and_back() -> None:
    """Test conversion GetParametersRes --> RecordSet --> GetParametersRes."""
    arrays = get_ndarrays()
    getparameteres_res = GetParametersRes(
        status=Status(code=Code(0), message=""),
        parameters=ndarrays_to_parameters(arrays),
    )

    getparameters_res_copy = deepcopy(getparameteres_res)

    recordset = getparameters_res_to_recordset(getparameteres_res)
    getparameteres_res_ = recordset_to_getparameters_res(recordset)

    assert getparameters_res_copy == getparameteres_res_


@pytest.mark.parametrize(
    "ins, convert_fn, task_type",
    [
        (_get_valid_fitins, partial(fit_ins_to_recordset, keep_input=False), "fit_ins"),
        (
            _get_valid_evaluateins,
            partial(evaluate_ins_to_recordset, keep_input=False),
            "evaluate_ins",
        ),
        (
            _get_valid_getpropertiesins,
            getproperties_ins_to_recordset,
            "get_properties_ins",
        ),
        (
            _get_valid_getparametersins,
            getparameters_ins_to_recordset,
            "get_parameters_ins",
        ),
    ],
)
def test_flowercontext_driver_to_client(
    ins: Union[FitIns, EvaluateIns, GetPropertiesIns, GetParametersIns],
    convert_fn: Union[
        Callable[[FitIns], RecordSet],
        Callable[[EvaluateIns], RecordSet],
        Callable[[GetPropertiesIns], RecordSet],
        Callable[[GetParametersIns], RecordSet],
    ],
    task_type: str,
) -> None:
    """."""
    f_context = FlowerContext(
        in_message=RecordSet(),
        out_message=convert_fn(ins()),
        local=RecordSet(),
        metadata=Metadata(
            run_id=0, task_id="", group_id="", ttl="", task_type=task_type
        ),
    )

    # TODO: embedd `f_context` in TaskIns

    # Construct FlowerContext from TaskIns
