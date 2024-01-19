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
"""RecordSet utilities."""


from typing import Dict, Mapping, OrderedDict, Tuple, Union, cast, get_args

from .configsrecord import ConfigsRecord
from .metricsrecord import MetricsRecord
from .parametersrecord import Array, ParametersRecord
from .recordset import RecordSet
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
    Parameters,
    Scalar,
    Status,
)


def parametersrecord_to_parameters(
    record: ParametersRecord, keep_input: bool = False
) -> Parameters:
    """Convert ParameterRecord to legacy Parameters.

    Warning: Because `Arrays` in `ParametersRecord` encode more information of the
    array-like or tensor-like data (e.g their datatype, shape) than `Parameters` it
    might not be possible to reconstruct such data structures from `Parameters` objects
    alone. Additional information or metadta must be provided from elsewhere.

    Parameters
    ----------
    record : ParametersRecord
        The record to be conveted into Parameters.
    keep_input : bool (default: False)
        A boolean indicating whether entries in the record should be deleted from the
        input dictionary immediately after adding them to the record.
    """
    parameters = Parameters(tensors=[], tensor_type="")

    for key in list(record.data.keys()):
        parameters.tensors.append(record.data[key].data)

        if not keep_input:
            del record.data[key]

    return parameters


def parameters_to_parametersrecord(
    parameters: Parameters, keep_input: bool = False
) -> ParametersRecord:
    """Convert legacy Parameters into a single ParametersRecord.

    Because there is no concept of names in the legacy Parameters, arbitrary keys will
    be used when constructing the ParametersRecord. Similarly, the shape and data type
    won't be recorded in the Array objects.

    Parameters
    ----------
    parameters : Parameters
        Parameters object to be represented as a ParametersRecord.
    keep_input : bool (default: False)
        A boolean indicating whether parameters should be deleted from the input
        Parameters object (i.e. a list of serialized NumPy arrays) immediately after
        adding them to the record.
    """
    tensor_type = parameters.tensor_type

    p_record = ParametersRecord()

    num_arrays = len(parameters.tensors)
    for idx in range(num_arrays):
        if keep_input:
            tensor = parameters.tensors[idx]
        else:
            tensor = parameters.tensors.pop(0)
        p_record.set_parameters(
            OrderedDict(
                {str(idx): Array(data=tensor, dtype="", stype=tensor_type, shape=[])}
            )
        )

    return p_record


def _check_mapping_from_scalar_to_metricsrecordstypes(
    scalar_config: Dict[str, Scalar],
) -> Dict[str, MetricsRecordValues]:
    """."""
    for value in scalar_config.values():
        if not isinstance(value, get_args(MetricsRecordValues)):
            raise TypeError(
                f"Supported types are {MetricsRecordValues}. "
                f"But you used type: {type(value)}"
            )
    return cast(Dict[str, MetricsRecordValues], scalar_config)


def _check_mapping_from_scalar_to_configsrecordstypes(
    scalar_config: Dict[str, Scalar],
) -> Dict[str, ConfigsRecordValues]:
    """."""
    for value in scalar_config.values():
        if not isinstance(value, get_args(ConfigsRecordValues)):
            raise TypeError(
                f"Supported types are {ConfigsRecordValues}. "
                f"But you used type: {type(value)}"
            )
    return cast(Dict[str, ConfigsRecordValues], scalar_config)


def _check_mapping_from_recordscalartype_to_scalar(
    record_data: Mapping[str, Union[ConfigsRecordValues, MetricsRecordValues]]
) -> Dict[str, Scalar]:
    """Check mapping `common.*RecordValues` into `common.Scalar` is possible."""
    for value in record_data.values():
        if not isinstance(value, get_args(Scalar)):
            raise TypeError(
                "There is not a 1:1 mapping between `common.Scalar` types and those "
                "supported in `common.ConfigsRecordValues` or "
                "`common.ConfigsRecordValues`. Consider casting your values to a type "
                "supported by the `common.RecordSet` infrastructure. "
                f"You used type: {type(value)}"
            )
    return cast(Dict[str, Scalar], record_data)


def _recordset_to_fit_or_evaluate_ins_components(
    recordset: RecordSet,
    ins_str: str,
    keep_input: bool,
) -> Tuple[Parameters, Dict[str, Scalar]]:
    """Derive Fit/Evaluate Ins from a RecordSet."""
    # get Array and construct Parameters
    parameters_record = recordset.get_parameters(f"{ins_str}.parameters")

    parameters = parametersrecord_to_parameters(
        parameters_record, keep_input=keep_input
    )

    # get config dict
    config_record = recordset.get_configs(f"{ins_str}.config")

    config_dict = _check_mapping_from_recordscalartype_to_scalar(config_record.data)

    return parameters, config_dict


def _fit_or_evaluate_ins_to_recordset(ins: Union[FitIns, EvaluateIns]) -> RecordSet:
    recordset = RecordSet()

    ins_str = "fitins" if isinstance(ins, FitIns) else "evaluateins"
    recordset.set_parameters(
        name=f"{ins_str}.parameters",
        record=parameters_to_parametersrecord(ins.parameters),
    )

    config = _check_mapping_from_scalar_to_configsrecordstypes(ins.config)
    recordset.set_configs(name=f"{ins_str}.config", record=ConfigsRecord(config))

    return recordset


def _embed_status_into_recordset(
    res_str: str, status: Status, recordset: RecordSet
) -> RecordSet:
    status_dict: Dict[str, ConfigsRecordValues] = {
        "code": int(status.code.value),
        "message": status.message,
    }
    recordset.set_configs(f"{res_str}.status", record=ConfigsRecord(status_dict))
    return recordset


def _extract_status_from_recordset(res_str: str, recordset: RecordSet) -> Status:
    status = recordset.get_metrics(f"{res_str}.status")
    code = cast(int, status.data["code"])
    return Status(code=Code(code), message=str(status.data["message"]))


def recordset_to_fit_ins(recordset: RecordSet, keep_input: bool) -> FitIns:
    """Derive FitIns from a RecordSet object."""
    parameters, config = _recordset_to_fit_or_evaluate_ins_components(
        recordset,
        ins_str="fitins",
        keep_input=keep_input,
    )

    return FitIns(parameters=parameters, config=config)


def fit_ins_to_recordset(fitins: FitIns) -> RecordSet:
    """Construct a RecordSet from a FitIns object."""
    return _fit_or_evaluate_ins_to_recordset(fitins)


def recordset_to_fit_res(recordset: RecordSet) -> FitRes:
    """Derive FitRes from a RecordSet object."""
    ins_str = "fitres"
    parameters = parametersrecord_to_parameters(
        recordset.get_parameters(f"{ins_str}.parameters")
    )

    num_examples = cast(
        int, recordset.get_metrics(f"{ins_str}.num_examples").data["num_exampes"]
    )
    metrics_record = recordset.get_metrics(f"{ins_str}.metrics")

    metrics = _check_mapping_from_recordscalartype_to_scalar(metrics_record.data)
    status = _extract_status_from_recordset(ins_str, recordset)

    return FitRes(
        status=status, parameters=parameters, num_examples=num_examples, metrics=metrics
    )


def fit_res_to_recordset(fitres: FitRes) -> RecordSet:
    """Construct a RecordSet from a FitRes object."""
    recordset = RecordSet()

    res_str = "fitres"

    metrics = _check_mapping_from_scalar_to_metricsrecordstypes(fitres.metrics)
    recordset.set_metrics(name=f"{res_str}.metrics", record=MetricsRecord(metrics))
    recordset.set_metrics(
        name=f"{res_str}.num_examples",
        record=MetricsRecord({"num_examples": fitres.num_examples}),
    )
    recordset.set_parameters(
        name=f"{res_str}.parameters",
        record=parameters_to_parametersrecord(fitres.parameters),
    )

    # status
    recordset = _embed_status_into_recordset(res_str, fitres.status, recordset)

    return recordset


def recordset_to_evaluate_ins(recordset: RecordSet, keep_input: bool) -> EvaluateIns:
    """Derive EvaluateIns from a RecordSet object."""
    parameters, config = _recordset_to_fit_or_evaluate_ins_components(
        recordset,
        ins_str="evaluateins",
        keep_input=keep_input,
    )

    return EvaluateIns(parameters=parameters, config=config)


def evaluate_ins_to_recordset(evaluateins: EvaluateIns) -> RecordSet:
    """Construct a RecordSet from a EvaluateIns object."""
    return _fit_or_evaluate_ins_to_recordset(evaluateins)


def recordset_to_evaluate_res(recordset: RecordSet) -> EvaluateRes:
    """Derive EvaluateRes from a RecordSet object."""
    ins_str = "evaluateres"

    loss = cast(int, recordset.get_metrics(f"{ins_str}.loss").data["loss"])

    num_examples = cast(
        int, recordset.get_metrics(f"{ins_str}.num_examples").data["num_exampes"]
    )
    metrics_record = recordset.get_metrics(f"{ins_str}.metrics")

    metrics = _check_mapping_from_recordscalartype_to_scalar(metrics_record.data)
    status = _extract_status_from_recordset(ins_str, recordset)

    return EvaluateRes(
        status=status, loss=loss, num_examples=num_examples, metrics=metrics
    )


def evaluate_res_to_recordset(evaluateres: EvaluateRes) -> RecordSet:
    """Construct a RecordSet from a EvaluateRes object."""
    recordset = RecordSet()

    res_str = "evaluateres"
    # loss
    recordset.set_metrics(
        name=f"{res_str}.loss",
        record=MetricsRecord({"loss": evaluateres.loss}),
    )

    # num_examples
    recordset.set_metrics(
        name=f"{res_str}.num_examples",
        record=MetricsRecord({"num_examples": evaluateres.num_examples}),
    )

    # metrics
    metrics = _check_mapping_from_scalar_to_metricsrecordstypes(evaluateres.metrics)
    recordset.set_metrics(name=f"{res_str}.metrics", record=MetricsRecord(metrics))

    # status
    recordset = _embed_status_into_recordset(
        f"{res_str}", evaluateres.status, recordset
    )

    return recordset


def recordset_to_getparameters_ins(recordset: RecordSet) -> GetParametersIns:
    """Derive GetParametersIns from a RecordSet object."""
    config_record = recordset.get_configs("getparametersins.config")

    config_dict = _check_mapping_from_recordscalartype_to_scalar(config_record.data)

    return GetParametersIns(config=config_dict)


def getparameters_ins_to_recordset(getparameters_ins: GetParametersIns) -> RecordSet:
    """Construct a RecordSet from a GetParametersIns object."""
    recordset = RecordSet()

    config = _check_mapping_from_scalar_to_configsrecordstypes(getparameters_ins.config)
    recordset.set_configs(name="getparametersins.config", record=ConfigsRecord(config))
    return recordset


def getparameters_res_to_recordset(getparametersres: GetParametersRes) -> RecordSet:
    """Construct a RecordSet from a GetParametersRes object."""
    recordset = RecordSet()
    parameters_record = parameters_to_parametersrecord(getparametersres.parameters)
    recordset.set_parameters("getparametersres.parameters", parameters_record)

    # status
    recordset = _embed_status_into_recordset(
        "getparametersres", getparametersres.status, recordset
    )

    return recordset


def recordset_to_getparameters_res(recordset: RecordSet) -> GetParametersRes:
    """Derive GetParametersRes from a RecordSet object."""
    res_str = "getparametersres"
    parameters = parametersrecord_to_parameters(
        recordset.get_parameters(f"{res_str}.parameters")
    )

    status = _extract_status_from_recordset(res_str, recordset)
    return GetParametersRes(status=status, parameters=parameters)


def recordset_to_getproperties_ins(recordset: RecordSet) -> GetPropertiesIns:
    """Derive GetPropertiesIns from a RecordSet object."""
    config_record = recordset.get_configs("getpropertiesins.config")
    config_dict = _check_mapping_from_recordscalartype_to_scalar(config_record.data)

    return GetPropertiesIns(config=config_dict)


def getproperties_ins_to_recordset(getpropertiesins: GetPropertiesIns) -> RecordSet:
    """Construct a RecordSet from a GetPropertiesRes object."""
    recordset = RecordSet()
    config_dict = _check_mapping_from_scalar_to_configsrecordstypes(
        getpropertiesins.config
    )
    recordset.set_configs(
        name="getpropertiesins.config", record=ConfigsRecord(config_dict)
    )
    return recordset


def recordset_to_getproperties_res(recordset: RecordSet) -> GetPropertiesRes:
    """Derive GetPropertiesRes from a RecordSet object."""
    res_str = "getpropertiesres"
    config_record = recordset.get_configs(f"{res_str}.config")
    properties = _check_mapping_from_recordscalartype_to_scalar(config_record.data)

    status = _extract_status_from_recordset(res_str, recordset=recordset)

    return GetPropertiesRes(status=status, properties=properties)


def getproperties_res_to_recordset(getpropertiesres: GetPropertiesRes) -> RecordSet:
    """Construct a RecordSet from a GetPropertiesRes object."""
    recordset = RecordSet()
    configs = _check_mapping_from_scalar_to_configsrecordstypes(
        getpropertiesres.properties
    )
    recordset.set_configs(
        name="getpropertiesres.properties", record=ConfigsRecord(configs)
    )
    # status
    recordset = _embed_status_into_recordset(
        "getpropertiesres", getpropertiesres.status, recordset
    )

    return recordset
