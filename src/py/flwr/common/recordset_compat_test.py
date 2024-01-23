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

from contextlib import nullcontext
from copy import deepcopy
from typing import Any, Dict

import numpy as np
import pytest

from .parameter import ndarrays_to_parameters
from .recordset_compat import (
    evaluate_ins_to_recordset,
    evaluate_res_to_recordset,
    fit_ins_to_recordset,
    fit_res_to_recordset,
    getparameters_ins_to_recordset,
    getparameters_res_to_recordset,
    getproperties_ins_to_recordset,
    getproperties_res_to_recordset,
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


def _get_valid_fitres_with_config(metrics: Dict[str, Scalar]) -> FitRes:
    """Returnn Valid parameters but potentially invalid config."""
    arrays = get_ndarrays()
    return FitRes(
        parameters=ndarrays_to_parameters(arrays),
        num_examples=1,
        status=Status(code=Code(0), message=""),
        metrics=metrics,
    )


def _get_valid_evaluateins() -> EvaluateIns:
    fit_ins = _get_valid_fitins()
    return EvaluateIns(parameters=fit_ins.parameters, config=fit_ins.config)


def _get_valid_evaluateres_with_config(metrics: Dict[str, Scalar]) -> EvaluateRes:
    """Return potentially invalid config."""
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
    fitres = _get_valid_fitres_with_config(metrics)

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
    evaluateres = _get_valid_evaluateres_with_config(metrics)

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
    getproperties_res = _get_valid_getpropertiesres()

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
    getparameteres_res = _get_valid_getparametersres()

    getparameters_res_copy = deepcopy(getparameteres_res)

    recordset = getparameters_res_to_recordset(getparameteres_res)
    getparameteres_res_ = recordset_to_getparameters_res(recordset)

    assert getparameters_res_copy == getparameteres_res_
