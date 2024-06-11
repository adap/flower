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
from typing import Literal, TypeVar, cast, overload

from ..constant import Record
from .configsrecord import ConfigsRecord
from .metricsrecord import MetricsRecord
from .parametersrecord import ParametersRecord
from .typeddict import TypedDict

T = TypeVar("T", ParametersRecord, MetricsRecord, ConfigsRecord)


class _Checker:
    def __init__(
        self,
        data: RecordSetData,
        type: type[ParametersRecord] | type[MetricsRecord] | type[ConfigsRecord],
    ) -> None:
        self.data = data
        self._type = type

    def check_key(self, key: str) -> None:
        data = self.data
        if not isinstance(key, str):
            raise TypeError(
                f"Expected `{str.__name__}`, but "
                f"received `{type(key).__name__}` for the key."
            )

        orig_value: ParametersRecord | MetricsRecord | ConfigsRecord | None = None
        if key in data.parameters_records:
            orig_value = data.parameters_records[key]
        elif key in data.metrics_records:
            orig_value = data.metrics_records[key]
        elif key in data.configs_records:
            orig_value = data.configs_records[key]

        if orig_value is not None and not isinstance(orig_value, self._type):
            raise KeyError(
                f"Key '{key}' is already associated with "
                f"a '{type(orig_value).__name__}', but a value "
                f"of type '{self._type.__name__}' was provided."
            )

    def check_value(
        self, value: ParametersRecord | MetricsRecord | ConfigsRecord
    ) -> None:
        if not isinstance(value, self._type):
            raise TypeError(
                f"Expected `{self._type.__name__}`, but received "
                f"`{type(value).__name__}` for the value."
            )


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
        params_checker = _Checker(self, ParametersRecord)
        metrics_checker = _Checker(self, MetricsRecord)
        configs_checker = _Checker(self, ConfigsRecord)
        self.parameters_records = TypedDict[str, ParametersRecord](
            params_checker.check_key, params_checker.check_value
        )
        self.metrics_records = TypedDict[str, MetricsRecord](
            metrics_checker.check_key, metrics_checker.check_value
        )
        self.configs_records = TypedDict[str, ConfigsRecord](
            configs_checker.check_key, configs_checker.check_value
        )
        if parameters_records is not None:
            self.parameters_records.update(parameters_records)
        if metrics_records is not None:
            self.metrics_records.update(metrics_records)
        if configs_records is not None:
            self.configs_records.update(configs_records)


class RecordSet:
    """RecordSet stores groups of parameters, metrics and configs."""

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

    @overload
    def __getitem__(self, key: Literal["p"]) -> ParametersRecord: ...

    @overload
    def __getitem__(self, key: Literal["m"]) -> MetricsRecord: ...

    @overload
    def __getitem__(self, key: Literal["c"]) -> ConfigsRecord: ...

    def __getitem__(self, key: str) -> ParametersRecord | MetricsRecord | ConfigsRecord:
        """Return the record for the specified key."""
        # Initialize default *Record
        data = cast(RecordSetData, self.__dict__["_data"])
        if key == Record.PARAMS and key not in data.parameters_records:
            data.parameters_records[key] = ParametersRecord()
        elif key == Record.METRICS and key not in data.metrics_records:
            data.metrics_records[key] = MetricsRecord()
        elif key == Record.CONFIGS and key not in data.configs_records:
            data.configs_records[key] = ConfigsRecord()

        # Return the record
        if key in data.parameters_records:
            return data.parameters_records[key]
        if key in data.metrics_records:
            return data.metrics_records[key]
        if key in data.configs_records:
            return data.configs_records[key]
        raise KeyError(key)

    def __setitem__(
        self, key: str, value: ParametersRecord | MetricsRecord | ConfigsRecord
    ) -> None:
        """Set the record for the specified key."""
        data = cast(RecordSetData, self.__dict__["_data"])
        if isinstance(value, ParametersRecord):
            data.parameters_records[key] = value
        elif isinstance(value, MetricsRecord):
            data.metrics_records[key] = value
        elif isinstance(value, ConfigsRecord):
            data.configs_records[key] = value
        else:
            raise TypeError(
                f"Expected {{`{ParametersRecord.__name__}`, "
                f"`{MetricsRecord.__name__}`, `{ConfigsRecord.__name__}`}}, "
                f"but received `{type(value).__name__}` for the value."
            )

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
