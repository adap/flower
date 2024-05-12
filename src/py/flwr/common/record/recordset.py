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


from dataclasses import dataclass
from typing import Dict, Optional, cast

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
        parameters_records: Optional[Dict[str, ParametersRecord]] = None,
        metrics_records: Optional[Dict[str, MetricsRecord]] = None,
        configs_records: Optional[Dict[str, ConfigsRecord]] = None,
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
    """RecordSet stores groups of parameters, metrics and configs."""

    def __init__(
        self,
        parameters_records: Optional[Dict[str, ParametersRecord]] = None,
        metrics_records: Optional[Dict[str, MetricsRecord]] = None,
        configs_records: Optional[Dict[str, ConfigsRecord]] = None,
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
