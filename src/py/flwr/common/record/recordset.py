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
"""RecordSet."""


from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from .configsrecord import ConfigsRecord
from .metricsrecord import MetricsRecord
from .parametersrecord import ParametersRecord
from .typeddict import TypedDict


@dataclass
class RecordSetData:
    """Inner data container for the RecordSet class."""

    parameters_records: TypedDict[str, ParametersRecord]
    metrics_records: TypedDict[str, MetricsRecord]
    configs_records: TypedDict[str, ConfigsRecord]

    def __init__(
        self,
        parameters_records: dict[str, ParametersRecord] | None = None,
        metrics_records: dict[str, MetricsRecord] | None = None,
        configs_records: dict[str, ConfigsRecord] | None = None,
    ) -> None:
        self.parameters_records = TypedDict[str, ParametersRecord](
            self._check_fn_str, self._check_fn_params
        )
        self.metrics_records = TypedDict[str, MetricsRecord](
            self._check_fn_str, self._check_fn_metrics
        )
        self.configs_records = TypedDict[str, ConfigsRecord](
            self._check_fn_str, self._check_fn_configs
        )
        if parameters_records is not None:
            self.parameters_records.update(parameters_records)
        if metrics_records is not None:
            self.metrics_records.update(metrics_records)
        if configs_records is not None:
            self.configs_records.update(configs_records)

    def _check_fn_str(self, key: str) -> None:
        if not isinstance(key, str):
            raise TypeError(
                f"Expected `{str.__name__}`, but "
                f"received `{type(key).__name__}` for the key."
            )

    def _check_fn_params(self, record: ParametersRecord) -> None:
        if not isinstance(record, ParametersRecord):
            raise TypeError(
                f"Expected `{ParametersRecord.__name__}`, but "
                f"received `{type(record).__name__}` for the value."
            )

    def _check_fn_metrics(self, record: MetricsRecord) -> None:
        if not isinstance(record, MetricsRecord):
            raise TypeError(
                f"Expected `{MetricsRecord.__name__}`, but "
                f"received `{type(record).__name__}` for the value."
            )

    def _check_fn_configs(self, record: ConfigsRecord) -> None:
        if not isinstance(record, ConfigsRecord):
            raise TypeError(
                f"Expected `{ConfigsRecord.__name__}`, but "
                f"received `{type(record).__name__}` for the value."
            )


class RecordSet:
    """RecordSet stores groups of parameters, metrics and configs.

    A :code:`RecordSet` is the unified mechanism by which parameters,
    metrics and configs can be either stored as part of a
    `flwr.common.Context <flwr.common.Context.html>`_ in your apps
    or communicated as part of a
    `flwr.common.Message <flwr.common.Message.html>`_ between your apps.

    Parameters
    ----------
    parameters_records : Optional[Dict[str, ParametersRecord]]
        A dictionary of :code:`ParametersRecords` that can be used to record
        and communicate model parameters and high-dimensional arrays.
    metrics_records : Optional[Dict[str, MetricsRecord]]
        A dictionary of :code:`MetricsRecord` that can be used to record
        and communicate scalar-valued metrics that are the result of performing
        and action, for example, by a :code:`ClientApp`.
    configs_records : Optional[Dict[str, ConfigsRecord]]
        A dictionary of :code:`ConfigsRecord` that can be used to record
        and communicate configuration values to an entity (e.g. to a
        :code:`ClientApp`)
        for it to adjust how an action is performed.

    Examples
    --------
    A :code:`RecordSet` can hold three types of records, each designed
    with an specific purpose. What is common to all of them is that they
    are Python dictionaries designed to ensure that each key-value pair
    adheres to specified data types.

    Let's see an example.

    >>>  from flwr.common import RecordSet
    >>>  from flwr.common import ConfigsRecord, MetricsRecord, ParametersRecord
    >>>
    >>>  # Let's begin with an empty record
    >>>  my_recordset = RecordSet()
    >>>
    >>>  # We can create a ConfigsRecord
    >>>  c_record = ConfigsRecord({"lr": 0.1, "batch-size": 128})
    >>>  # Adding it to the record_set would look like this
    >>>  my_recordset.configs_records["my_config"] = c_record
    >>>
    >>>  # We can create a MetricsRecord following a similar process
    >>>  m_record = MetricsRecord({"accuracy": 0.93, "losses": [0.23, 0.1]})
    >>>  # Adding it to the record_set would look like this
    >>>  my_recordset.metrics_records["my_metrics"] = m_record

    Adding a :code:`ParametersRecord` follows the same steps as above but first,
    the array needs to be serialized and represented as a :code:`flwr.common.Array`.
    If the array is a :code:`NumPy` array, you can use the built-in utility function
    `array_from_numpy <flwr.common.array_from_numpy.html>`_. It is often possible to
    convert an array first to :code:`NumPy` and then use the aforementioned function.

    >>>  from flwr.common import array_from_numpy
    >>>  # Creating a ParametersRecord would look like this
    >>>  arr_np = np.random.randn(3, 3)
    >>>
    >>>  # You can use the built-in tool to serialize the array
    >>>  arr = array_from_numpy(arr_np)
    >>>
    >>>  # Finally, create the record
    >>>  p_record = ParametersRecord({"my_array": arr})
    >>>
    >>>  # Adding it to the record_set would look like this
    >>>  my_recordset.configs_records["my_config"] = c_record

    For additional examples on how to construct each of the records types shown
    above, please refer to the documentation for :code:`ConfigsRecord`,
    :code:`MetricsRecord` and :code:`ParametersRecord`.
    """

    def __init__(
        self,
        parameters_records: dict[str, ParametersRecord] | None = None,
        metrics_records: dict[str, MetricsRecord] | None = None,
        configs_records: dict[str, ConfigsRecord] | None = None,
    ) -> None:
        data = RecordSetData(
            parameters_records=parameters_records,
            metrics_records=metrics_records,
            configs_records=configs_records,
        )
        self.__dict__["_data"] = data

    @property
    def parameters_records(self) -> TypedDict[str, ParametersRecord]:
        """Dictionary holding ParametersRecord instances."""
        data = cast(RecordSetData, self.__dict__["_data"])
        return data.parameters_records

    @property
    def metrics_records(self) -> TypedDict[str, MetricsRecord]:
        """Dictionary holding MetricsRecord instances."""
        data = cast(RecordSetData, self.__dict__["_data"])
        return data.metrics_records

    @property
    def configs_records(self) -> TypedDict[str, ConfigsRecord]:
        """Dictionary holding ConfigsRecord instances."""
        data = cast(RecordSetData, self.__dict__["_data"])
        return data.configs_records

    def __repr__(self) -> str:
        """Return a string representation of this instance."""
        flds = ("parameters_records", "metrics_records", "configs_records")
        view = ", ".join([f"{fld}={getattr(self, fld)!r}" for fld in flds])
        return f"{self.__class__.__qualname__}({view})"

    def __eq__(self, other: object) -> bool:
        """Compare two instances of the class."""
        if not isinstance(other, self.__class__):
            raise NotImplementedError
        return self.__dict__ == other.__dict__
