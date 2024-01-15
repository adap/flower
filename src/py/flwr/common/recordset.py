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
"""RecordSet."""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, cast

from .typing import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Parameters,
    Scalar,
    Status,
)


def _check(allowed_types: Any, element: Any) -> Tuple[bool, str]:
    """Check if passed element is of allowed type."""
    msg = ""
    check = isinstance(element, allowed_types)
    if not check:
        msg = f"It must be of type `{allowed_types}` but got `{type(element)}`"

    return check, msg


class Tensor:
    """Tensor type."""

    def __init__(self) -> None:
        self.data: List[bytes]
        self.shape: List[int]
        self.dtype: str  # tbd
        self.ref: str  # future functionality


class ParameterRecord(Dict[str, Tensor]):
    """Parameter record."""

    def __setitem__(self, key: str, value: Tensor) -> None:
        """Set item after key and value checks."""
        check, mssg = _check(str, key)
        if not check:
            raise TypeError(f"Key `{key}` is of invalid key type. {mssg}")
        check, mssg = _check(Tensor, value)
        if not check:
            raise TypeError(f"Value for key `{key}` is of invalid value type. {mssg}")
        super().__setitem__(key, value)


# TODO: MetricsRecord and ConfigsRecord should support type Value instead of just the set of types in Scalar
class MetricsRecord(Dict[str, Scalar]):
    """Metrics record."""

    def __setitem__(self, key: str, value: Scalar) -> None:
        """Set item after key and value checks."""
        check, mssg = _check(str, key)
        if not check:
            raise TypeError(f"Key `{key}` is of invalid key type. {mssg}")
        check, mssg = _check(Scalar, value)
        if not check:
            raise TypeError(f"Value for key `{key}` is of invalid value type. {mssg}")
        super().__setitem__(key, value)


class ConfigsRecord(Dict[str, Scalar]):
    """Config record."""

    def __setitem__(self, key: str, value: Scalar) -> None:
        """Set item after key and value checks."""
        check, mssg = _check(str, key)
        if not check:
            raise TypeError(f"Key `{key}` is of invalid key type. {mssg}")
        check, mssg = _check(Scalar, value)
        if not check:
            raise TypeError(f"Value for key `{key}` is of invalid value type. {mssg}")
        super().__setitem__(key, value)


@dataclass
class RecordSet:
    """Definition of RecordSet."""

    parameters: Dict[str, ParameterRecord] = {}
    metrics: Dict[str, MetricsRecord] = {}
    configs: Dict[str, ConfigsRecord] = {}

    def set_parameters(self, name: str, record: ParameterRecord) -> None:
        """Add a ParameterRecord."""
        self.parameters[name] = record

    def get_parameters(self, name: str) -> ParameterRecord:
        """Get a ParameterRecord."""
        return self.parameters[name]

    def set_metrics(self, name: str, record: MetricsRecord) -> None:
        """Add a MetricsRecord."""
        self.metrics[name] = record

    def get_metrics(self, name: str) -> MetricsRecord:
        """Get a MetricsRecord."""
        return self.metrics[name]

    def set_configs(self, name: str, record: ConfigsRecord) -> None:
        """Add a ConfigsRecord."""
        self.configs[name] = record

    def get_configs(self, name: str) -> ConfigsRecord:
        """Get a ConfigsRecord."""
        return self.configs[name]



################################## Fit

def fit_ins_to_recordset(fit_ins: FitIns) -> RecordSet:
    """."""
    r_set = RecordSet()

    tensor = Tensor()
    tensor.data = fit_ins.parameters.tensors
    tensor.dtype = fit_ins.parameters.tensor_type

    r_set.set_parameters(name="fitins", record=ParameterRecord({"parameters": tensor}))
    r_set.set_configs(name="fitins.config", record=ConfigsRecord(fit_ins.config))
    return r_set


def recordset_to_fit_ins(recordset: RecordSet) -> FitIns:
    """."""
    tensors = recordset.get_parameters("fitins")["parameters"]
    return FitIns(
        parameters=Parameters(tensors=tensors.data, tensor_type=tensors.dtype),
        config=recordset.get_configs(name="fitins.config"),
    )


def fit_res_to_recordset(fit_res: FitRes) -> RecordSet:
    """."""
    r_set = RecordSet()

    tensor = Tensor()
    tensor.data = fit_res.parameters.tensors
    tensor.dtype = fit_res.parameters.tensor_type
    r_set.set_parameters(name="fitres", record=ParameterRecord({"parameters": tensor}))

    r_set.set_metrics(
        name="fitres", record=MetricsRecord({"num_examples": fit_res.num_examples})
    )
    r_set.set_metrics(name="fitres.metrics", record=MetricsRecord(fit_res.metrics))
    r_set.set_metrics(
        name="fitres.status",
        record=MetricsRecord(
            {"code": int(fit_res.status.code.value), "message": fit_res.status.message}
        ),
    )

    return r_set


def recordset_to_fit_res(recordset: RecordSet) -> FitRes:
    """."""
    tensors = recordset.get_parameters("fitres")["parameters"]
    status = recordset.get_metrics("fitres.status")
    num_examples = recordset.get_metrics("fitres")["num_examples"]
    return FitRes(
        status=Status(*status),
        parameters=Parameters(tensors=tensors.data, tensor_type=tensors.dtype),
        num_examples=cast(int, num_examples),
        metrics=recordset.get_metrics("fitres.metrics"),
    )


################################## Evaluate


def evaluate_ins_to_recordset(evaluate_ins: EvaluateIns) -> RecordSet:
    """."""
    r_set = RecordSet()
    tensor = Tensor()
    tensor.data = evaluate_ins.parameters.tensors
    tensor.dtype = evaluate_ins.parameters.tensor_type

    r_set.set_parameters(
        name="evaluateins", record=ParameterRecord({"parameters": tensor})
    )
    r_set.set_configs(
        name="evaluateins.config", record=ConfigsRecord(evaluate_ins.config)
    )
    return r_set


def recordset_to_evaluate_ins(recordset: RecordSet) -> EvaluateIns:
    """."""
    tensors = recordset.get_parameters("evaluateins")["parameters"]
    return EvaluateIns(
        parameters=Parameters(tensors=tensors.data, tensor_type=tensors.dtype),
        config=recordset.get_configs(name="evaluateins.config"),
    )


def evaluate_res_to_recordset(evaluate_res: EvaluateRes) -> RecordSet:
    """."""
    r_set = RecordSet()

    r_set.set_metrics(
        name="evaluateres", record=MetricsRecord({"loss": evaluate_res.loss})
    )
    r_set.set_metrics(
        name="evaluateres",
        record=MetricsRecord({"num_examples": evaluate_res.num_examples}),
    )
    r_set.set_metrics(
        name="evaluateres.metrics", record=MetricsRecord(evaluate_res.metrics)
    )
    r_set.set_metrics(
        name="evaluateres.status",
        record=MetricsRecord(
            {
                "code": int(evaluate_res.status.code.value),
                "message": evaluate_res.status.message,
            }
        ),
    )

    return r_set


def recordset_to_evaluate_res(recordset: RecordSet) -> EvaluateRes:
    """."""
    status = recordset.get_metrics("evaluateres.status")
    num_examples = recordset.get_metrics("evaluateres")["num_examples"]
    loss = recordset.get_metrics("evaluateres")["loss"]
    return EvaluateRes(
        status=Status(*status),
        loss=cast(float, loss),
        num_examples=cast(int, num_examples),
        metrics=recordset.get_metrics("evaluateres.metrics"),
    )


################################## GetParameters


def getparameters_ins_to_recordset(getparameters_ins: GetParametersIns) -> RecordSet:
    """."""
    r_set = RecordSet()
    r_set.set_configs(
        name="getparametersins.config", record=ConfigsRecord(getparameters_ins.config)
    )
    return r_set


def recordset_to_getparameters_ins(recordset: RecordSet) -> GetParametersIns:
    """."""
    return GetParametersIns(config=recordset.get_configs("getparametersins.config"))


def getparameters_res_to_recordset(getparameters_res: GetParametersRes) -> RecordSet:
    """."""
    r_set = RecordSet()

    tensor = Tensor()
    tensor.data = getparameters_res.parameters.tensors
    tensor.dtype = getparameters_res.parameters.tensor_type

    r_set.set_parameters(
        name="getparametersres", record=ParameterRecord({"parameters": tensor})
    )

    r_set.set_metrics(
        name="getparametersres.status",
        record=MetricsRecord(
            {
                "code": int(getparameters_res.status.code.value),
                "message": getparameters_res.status.message,
            }
        ),
    )

    return r_set


def recordset_to_getparameters_res(recordset: RecordSet) -> GetParametersRes:
    """."""
    tensors = recordset.get_parameters("getparametersres")["parameters"]
    status = recordset.get_metrics("getparametersres.status")
    return GetParametersRes(
        status=Status(*status),
        parameters=Parameters(tensors=tensors.data, tensor_type=tensors.dtype),
    )


################################## GetProperties


def getproperties_ins_to_recordset(getproperties_ins: GetPropertiesIns) -> RecordSet:
    """."""
    r_set = RecordSet()
    r_set.set_configs(
        name="getpropertiesins.config", record=ConfigsRecord(getproperties_ins.config)
    )
    return r_set


def recordset_to_getproperties_ins(recordset: RecordSet) -> GetPropertiesIns:
    """."""
    return GetPropertiesIns(config=recordset.get_configs("getpropertiesins.config"))


def getproperties_res_to_recordset(getproperties_res: GetPropertiesRes) -> RecordSet:
    """."""
    r_set = RecordSet()
    r_set.set_metrics(
        name="getpropertiesres.status",
        record=MetricsRecord(
            {
                "code": int(getproperties_res.status.code.value),
                "message": getproperties_res.status.message,
            }
        ),
    )

    r_set.set_metrics(
        name="getproperties_res.properties",
        record=MetricsRecord(getproperties_res.properties),
    )

    return r_set


def recordset_to_getproperties_res(recordset: RecordSet) -> GetPropertiesRes:
    """."""
    status = recordset.get_metrics("getproperties_res.status")
    return GetPropertiesRes(
        status=Status(*status),
        properties=recordset.get_metrics("getproperties_res.properties"),
    )
