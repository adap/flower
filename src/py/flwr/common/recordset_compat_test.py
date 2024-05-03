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
"""RecordSet from legacy messages tests."""

from copy import deepcopy
from typing import Callable, Dict

import numpy as np
import pytest

from .parameter import ndarrays_to_parameters
from .recordset_compat import (
    evaluateins_to_recordset,
    evaluateres_to_recordset,
    fitins_to_recordset,
    fitres_to_recordset,
    getparametersins_to_recordset,
    getparametersres_to_recordset,
    getpropertiesins_to_recordset,
    getpropertiesres_to_recordset,
    recordset_to_evaluateins,
    recordset_to_evaluateres,
    recordset_to_fitins,
    recordset_to_fitres,
    recordset_to_getparametersins,
    recordset_to_getparametersres,
    recordset_to_getpropertiesins,
    recordset_to_getpropertiesres,
)
from .typing import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    NDArrays,
    Scalar,
    Status,
)


def get_ndarrays() -> NDArrays:
    """Return list of NumPy arrays."""
    arr1 = np.array([[1.0, 2.0], [3.0, 4], [5.0, 6.0]])
    arr2 = np.eye(2, 7, 3)

    return [arr1, arr2]


##################################################
#  Testing conversion: *Ins --> RecordSet --> *Ins
#  Testing conversion: *Res <-- RecordSet <-- *Res
##################################################


def _get_valid_fitins() -> FitIns:
    arrays = get_ndarrays()
    return FitIns(parameters=ndarrays_to_parameters(arrays), config={"a": 1.0, "b": 0})


def _get_valid_fitres() -> FitRes:
    """Returnn Valid parameters but potentially invalid config."""
    arrays = get_ndarrays()
    metrics: Dict[str, Scalar] = {"a": 1.0, "b": 0}
    return FitRes(
        parameters=ndarrays_to_parameters(arrays),
        num_examples=1,
        status=Status(code=Code(0), message=""),
        metrics=metrics,
    )


def _get_valid_evaluateins() -> EvaluateIns:
    fit_ins = _get_valid_fitins()
    return EvaluateIns(parameters=fit_ins.parameters, config=fit_ins.config)


def _get_valid_evaluateres() -> EvaluateRes:
    """Return potentially invalid config."""
    metrics: Dict[str, Scalar] = {"a": 1.0, "b": 0}
    return EvaluateRes(
        num_examples=1,
        loss=0.1,
        status=Status(code=Code(0), message=""),
        metrics=metrics,
    )


def _get_valid_getparametersins() -> GetParametersIns:
    config_dict: Dict[str, Scalar] = {
        "a": 1.0,
        "b": 3,
        "c": True,
    }  # valid since both Ins/Res communicate over ConfigsRecord

    return GetParametersIns(config_dict)


def _get_valid_getparametersres() -> GetParametersRes:
    arrays = get_ndarrays()
    return GetParametersRes(
        status=Status(code=Code(0), message=""),
        parameters=ndarrays_to_parameters(arrays),
    )


def _get_valid_getpropertiesins() -> GetPropertiesIns:
    getparamsins = _get_valid_getparametersins()
    return GetPropertiesIns(config=getparamsins.config)


def _get_valid_getpropertiesres() -> GetPropertiesRes:
    config_dict: Dict[str, Scalar] = {
        "a": 1.0,
        "b": 3,
        "c": True,
    }  # valid since both Ins/Res communicate over ConfigsRecord

    return GetPropertiesRes(
        status=Status(code=Code(0), message=""), properties=config_dict
    )


@pytest.mark.parametrize(
    "keep_input, validate_freed_fn",
    [
        (
            False,
            lambda x, x_copy, y: len(x.parameters.tensors) == 0 and x_copy == y,
        ),  # check tensors were freed
        (
            True,
            lambda x, x_copy, y: x == y,
        ),
    ],
)
def test_fitins_to_recordset_and_back(
    keep_input: bool, validate_freed_fn: Callable[[FitIns, FitIns, FitIns], bool]
) -> None:
    """Test conversion FitIns --> RecordSet --> FitIns."""
    fitins = _get_valid_fitins()

    fitins_copy = deepcopy(fitins)

    recordset = fitins_to_recordset(fitins, keep_input=keep_input)

    fitins_ = recordset_to_fitins(recordset, keep_input=keep_input)

    assert validate_freed_fn(fitins, fitins_copy, fitins_)


@pytest.mark.parametrize(
    "keep_input, validate_freed_fn",
    [
        (
            False,
            lambda x, x_copy, y: len(x.parameters.tensors) == 0 and x_copy == y,
        ),  # check tensors were freed
        (
            True,
            lambda x, x_copy, y: x == y,
        ),
    ],
)
def test_fitres_to_recordset_and_back(
    keep_input: bool, validate_freed_fn: Callable[[FitRes, FitRes, FitRes], bool]
) -> None:
    """Test conversion FitRes --> RecordSet --> FitRes."""
    fitres = _get_valid_fitres()

    fitres_copy = deepcopy(fitres)

    recordset = fitres_to_recordset(fitres, keep_input=keep_input)
    fitres_ = recordset_to_fitres(recordset, keep_input=keep_input)

    assert validate_freed_fn(fitres, fitres_copy, fitres_)


@pytest.mark.parametrize(
    "keep_input, validate_freed_fn",
    [
        (
            False,
            lambda x, x_copy, y: len(x.parameters.tensors) == 0 and x_copy == y,
        ),  # check tensors were freed
        (
            True,
            lambda x, x_copy, y: x == y,
        ),
    ],
)
def test_evaluateins_to_recordset_and_back(
    keep_input: bool,
    validate_freed_fn: Callable[[EvaluateIns, EvaluateIns, EvaluateIns], bool],
) -> None:
    """Test conversion EvaluateIns --> RecordSet --> EvaluateIns."""
    evaluateins = _get_valid_evaluateins()

    evaluateins_copy = deepcopy(evaluateins)

    recordset = evaluateins_to_recordset(evaluateins, keep_input=keep_input)

    evaluateins_ = recordset_to_evaluateins(recordset, keep_input=keep_input)

    assert validate_freed_fn(evaluateins, evaluateins_copy, evaluateins_)


def test_evaluateres_to_recordset_and_back() -> None:
    """Test conversion EvaluateRes --> RecordSet --> EvaluateRes."""
    evaluateres = _get_valid_evaluateres()

    evaluateres_copy = deepcopy(evaluateres)

    recordset = evaluateres_to_recordset(evaluateres)
    evaluateres_ = recordset_to_evaluateres(recordset)

    assert evaluateres_copy == evaluateres_


def test_get_properties_ins_to_recordset_and_back() -> None:
    """Test conversion GetPropertiesIns --> RecordSet --> GetPropertiesIns."""
    getproperties_ins = _get_valid_getpropertiesins()

    getproperties_ins_copy = deepcopy(getproperties_ins)

    recordset = getpropertiesins_to_recordset(getproperties_ins)
    getproperties_ins_ = recordset_to_getpropertiesins(recordset)

    assert getproperties_ins_copy == getproperties_ins_


def test_get_properties_res_to_recordset_and_back() -> None:
    """Test conversion GetPropertiesRes --> RecordSet --> GetPropertiesRes."""
    getproperties_res = _get_valid_getpropertiesres()

    getproperties_res_copy = deepcopy(getproperties_res)

    recordset = getpropertiesres_to_recordset(getproperties_res)
    getproperties_res_ = recordset_to_getpropertiesres(recordset)

    assert getproperties_res_copy == getproperties_res_


def test_get_parameters_ins_to_recordset_and_back() -> None:
    """Test conversion GetParametersIns --> RecordSet --> GetParametersIns."""
    getparameters_ins = _get_valid_getparametersins()

    getparameters_ins_copy = deepcopy(getparameters_ins)

    recordset = getparametersins_to_recordset(getparameters_ins)
    getparameters_ins_ = recordset_to_getparametersins(recordset)

    assert getparameters_ins_copy == getparameters_ins_


@pytest.mark.parametrize(
    "keep_input, validate_freed_fn",
    [
        (
            False,
            lambda x, x_copy, y: len(x.parameters.tensors) == 0 and x_copy == y,
        ),  # check tensors were freed
        (
            True,
            lambda x, x_copy, y: x == y,
        ),
    ],
)
def test_get_parameters_res_to_recordset_and_back(
    keep_input: bool,
    validate_freed_fn: Callable[
        [GetParametersRes, GetParametersRes, GetParametersRes], bool
    ],
) -> None:
    """Test conversion GetParametersRes --> RecordSet --> GetParametersRes."""
    getparameteres_res = _get_valid_getparametersres()

    getparameters_res_copy = deepcopy(getparameteres_res)

    recordset = getparametersres_to_recordset(getparameteres_res, keep_input=keep_input)
    getparameteres_res_ = recordset_to_getparametersres(
        recordset, keep_input=keep_input
    )

    assert validate_freed_fn(
        getparameteres_res, getparameters_res_copy, getparameteres_res_
    )
