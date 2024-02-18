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
from typing import Callable, Type, TypeVar

from .configsrecord import ConfigsRecord
from .metricsrecord import MetricsRecord
from .parametersrecord import ParametersRecord
from .typeddict import TypedDict

T = TypeVar("T")


@dataclass
class RecordSet:
    """Enhanced RecordSet with a unified and Pythonic interface."""

    _parameters: TypedDict[str, ParametersRecord]
    _metrics: TypedDict[str, MetricsRecord]
    _configs: TypedDict[str, ConfigsRecord]

    def __init__(self):
        def _get_check_fn(__t: Type[T]) -> Callable[[T], None]:
            def _check_fn(__v: T) -> None:
                if not isinstance(__v, __t):
                    raise TypeError(f"Key must be of type str. You passed {type(__v)}.")

            return _check_fn

        self._parameters = TypedDict[str, ParametersRecord](
            _get_check_fn(str), _get_check_fn(ParametersRecord)
        )
        self._metrics = TypedDict[str, MetricsRecord](
            _get_check_fn(str), _get_check_fn(MetricsRecord)
        )
        self._configs = TypedDict[str, ConfigsRecord](
            _get_check_fn(str), _get_check_fn(ConfigsRecord)
        )

    @property
    def parameters_dict(self) -> TypedDict[str, ParametersRecord]:
        """Dictionary of ParametersRecord."""
        return self._parameters

    @property
    def metrics_dict(self) -> TypedDict[str, MetricsRecord]:
        """Dictionary of MetricsRecord."""
        return self._metrics

    @property
    def configs_dict(self) -> TypedDict[str, ConfigsRecord]:
        """Dictionary of ConfigsRecord."""
        return self._configs
