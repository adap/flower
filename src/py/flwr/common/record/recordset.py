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
from typing import Callable, Dict, Optional, Type, TypeVar

from .configsrecord import ConfigsRecord
from .metricsrecord import MetricsRecord
from .parametersrecord import ParametersRecord
from .typeddict import TypedDict

T = TypeVar("T")


@dataclass
class RecordSet:
    """RecordSet stores groups of parameters, metrics and configs."""

    _parameters_records: TypedDict[str, ParametersRecord]
    _metrics_records: TypedDict[str, MetricsRecord]
    _configs_records: TypedDict[str, ConfigsRecord]

    def __init__(
        self,
        parameters_records: Optional[Dict[str, ParametersRecord]] = None,
        metrics_records: Optional[Dict[str, MetricsRecord]] = None,
        configs_records: Optional[Dict[str, ConfigsRecord]] = None,
    ) -> None:
        def _get_check_fn(__t: Type[T]) -> Callable[[T], None]:
            def _check_fn(__v: T) -> None:
                if not isinstance(__v, __t):
                    raise TypeError(f"Expected `{__t}`, but `{type(__v)}` was passed.")

            return _check_fn

        self._parameters_records = TypedDict[str, ParametersRecord](
            _get_check_fn(str), _get_check_fn(ParametersRecord)
        )
        self._metrics_records = TypedDict[str, MetricsRecord](
            _get_check_fn(str), _get_check_fn(MetricsRecord)
        )
        self._configs_records = TypedDict[str, ConfigsRecord](
            _get_check_fn(str), _get_check_fn(ConfigsRecord)
        )
        if parameters_records is not None:
            self._parameters_records.update(parameters_records)
        if metrics_records is not None:
            self._metrics_records.update(metrics_records)
        if configs_records is not None:
            self._configs_records.update(configs_records)

    @property
    def parameters_records(self) -> TypedDict[str, ParametersRecord]:
        """Dictionary holding ParametersRecord instances."""
        return self._parameters_records

    @property
    def metrics_records(self) -> TypedDict[str, MetricsRecord]:
        """Dictionary holding MetricsRecord instances."""
        return self._metrics_records

    @property
    def configs_records(self) -> TypedDict[str, ConfigsRecord]:
        """Dictionary holding ConfigsRecord instances."""
        return self._configs_records
