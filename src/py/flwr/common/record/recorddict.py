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

from logging import WARN
from textwrap import indent
from typing import TypeVar, Union, cast

from ..logger import log
from .configsrecord import ConfigsRecord
from .metricsrecord import MetricsRecord
from .parametersrecord import ParametersRecord
from .typeddict import TypedDict

RecordType = Union[ParametersRecord, MetricsRecord, ConfigsRecord]

T = TypeVar("T")


def _check_key(key: str) -> None:
    if not isinstance(key, str):
        raise TypeError(
            f"Expected `{str.__name__}`, but "
            f"received `{type(key).__name__}` for the key."
        )


def _check_value(value: RecordType) -> None:
    if not isinstance(value, (ParametersRecord, MetricsRecord, ConfigsRecord)):
        raise TypeError(
            f"Expected `{ParametersRecord.__name__}`, `{MetricsRecord.__name__}`, "
            f"or `{ConfigsRecord.__name__}` but received "
            f"`{type(value).__name__}` for the value."
        )


class _SyncedDict(TypedDict[str, T]):
    """A synchronized dictionary that mirrors changes to an underlying RecordSet.

    This dictionary ensures that any modifications (set or delete operations)
    are automatically reflected in the associated `RecordSet`. Only values of
    the specified `allowed_type` are permitted.
    """

    def __init__(self, ref_recordset: RecordSet, allowed_type: type[T]) -> None:
        if not issubclass(
            allowed_type, (ParametersRecord, MetricsRecord, ConfigsRecord)
        ):
            raise TypeError(f"{allowed_type} is not a valid type.")
        super().__init__(_check_key, self.check_value)
        self.recordset = ref_recordset
        self.allowed_type = allowed_type

    def __setitem__(self, key: str, value: T) -> None:
        super().__setitem__(key, value)
        self.recordset[key] = cast(RecordType, value)

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        del self.recordset[key]

    def check_value(self, value: T) -> None:
        """Check if value is of expected type."""
        if not isinstance(value, self.allowed_type):
            raise TypeError(
                f"Expected `{self.allowed_type.__name__}`, but "
                f"received `{type(value).__name__}` for the value."
            )


class RecordSet(TypedDict[str, RecordType]):
    """RecordSet stores groups of parameters, metrics and configs.

    A :class:`RecordSet` is the unified mechanism by which parameters,
    metrics and configs can be either stored as part of a :class:`Context`
    in your apps or communicated as part of a :class:`Message` between
    your apps.

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
    >>>  my_recordset["my_config"] = c_record
    >>>
    >>>  # We can create a MetricsRecord following a similar process
    >>>  m_record = MetricsRecord({"accuracy": 0.93, "losses": [0.23, 0.1]})
    >>>  # Adding it to the record_set would look like this
    >>>  my_recordset["my_metrics"] = m_record

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
    >>>  my_recordset["my_parameters"] = p_record

    For additional examples on how to construct each of the records types shown
    above, please refer to the documentation for :code:`ConfigsRecord`,
    :code:`MetricsRecord` and :code:`ParametersRecord`.
    """

    def __init__(self, records: dict[str, RecordType] | None = None) -> None:
        super().__init__(_check_key, _check_value)
        if records is not None:
            for key, record in records.items():
                self[key] = record

    @property
    def parameters_records(self) -> TypedDict[str, ParametersRecord]:
        """Dictionary holding only ParametersRecord instances."""
        synced_dict = _SyncedDict[ParametersRecord](self, ParametersRecord)
        for key, record in self.items():
            if isinstance(record, ParametersRecord):
                synced_dict[key] = record
        return synced_dict

    @property
    def metrics_records(self) -> TypedDict[str, MetricsRecord]:
        """Dictionary holding only MetricsRecord instances."""
        synced_dict = _SyncedDict[MetricsRecord](self, MetricsRecord)
        for key, record in self.items():
            if isinstance(record, MetricsRecord):
                synced_dict[key] = record
        return synced_dict

    @property
    def configs_records(self) -> TypedDict[str, ConfigsRecord]:
        """Dictionary holding only ConfigsRecord instances."""
        synced_dict = _SyncedDict[ConfigsRecord](self, ConfigsRecord)
        for key, record in self.items():
            if isinstance(record, ConfigsRecord):
                synced_dict[key] = record
        return synced_dict

    def __repr__(self) -> str:
        """Return a string representation of this instance."""
        flds = ("parameters_records", "metrics_records", "configs_records")
        fld_views = [f"{fld}={dict(getattr(self, fld))!r}" for fld in flds]
        view = indent(",\n".join(fld_views), "  ")
        return f"{self.__class__.__qualname__}(\n{view}\n)"

    def __setitem__(self, key: str, value: RecordType) -> None:
        """Set the given key to the given value after type checking."""
        original_value = self.get(key, None)
        super().__setitem__(key, value)
        if original_value is not None and not isinstance(value, type(original_value)):
            log(
                WARN,
                "Key '%s' was overwritten: record of type `%s` replaced with type `%s`",
                key,
                type(original_value).__name__,
                type(value).__name__,
            )
