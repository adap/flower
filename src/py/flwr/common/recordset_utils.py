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


from typing import OrderedDict

from .parametersrecord import Array, ParametersRecord
from .typing import Parameters


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
