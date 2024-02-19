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
from typing import Callable, Dict, Type, TypeVar

from .configsrecord import ConfigsRecord
from .metricsrecord import MetricsRecord
from .parametersrecord import ParametersRecord
from .typeddict import TypedDict

T = TypeVar("T")
DEFAULT_PARAMETERS = "parameters:builtin:default"
DEFAULT_METRICS = "metrics:builtin:default"
DEFAULT_CONFIGS = "configs:builtin:default"


@dataclass
class RecordSet:
    """Enhanced RecordSet with a unified and Pythonic interface."""

    _parameters_dict: TypedDict[str, ParametersRecord]
    _metrics_dict: TypedDict[str, MetricsRecord]
    _configs_dict: TypedDict[str, ConfigsRecord]

    def __init__(
        self,
        parameters_dict: Dict[str, ParametersRecord] | None = None,
        metrics_dict: Dict[str, MetricsRecord] | None = None,
        configs_dict: Dict[str, ConfigsRecord] | None = None,
    ) -> None:
        def _get_check_fn(__t: Type[T]) -> Callable[[T], None]:
            def _check_fn(__v: T) -> None:
                if not isinstance(__v, __t):
                    raise TypeError(f"Key must be of type str. You passed {type(__v)}.")

            return _check_fn

        self._parameters_dict = TypedDict[str, ParametersRecord](
            _get_check_fn(str), _get_check_fn(ParametersRecord)
        )
        self._metrics_dict = TypedDict[str, MetricsRecord](
            _get_check_fn(str), _get_check_fn(MetricsRecord)
        )
        self._configs_dict = TypedDict[str, ConfigsRecord](
            _get_check_fn(str), _get_check_fn(ConfigsRecord)
        )
        if parameters_dict is not None:
            self._parameters_dict.update(parameters_dict)
        if metrics_dict is not None:
            self._metrics_dict.update(metrics_dict)
        if configs_dict is not None:
            self._configs_dict.update(configs_dict)

    @property
    def parameters_dict(self) -> TypedDict[str, ParametersRecord]:
        """Dictionary of ParametersRecord."""
        return self._parameters_dict

    @property
    def metrics_dict(self) -> TypedDict[str, MetricsRecord]:
        """Dictionary of MetricsRecord."""
        return self._metrics_dict

    @property
    def configs_dict(self) -> TypedDict[str, ConfigsRecord]:
        """Dictionary of ConfigsRecord."""
        return self._configs_dict

    @property
    def default_parameters(self) -> ParametersRecord:
        """Default ParametersRecord."""
        if DEFAULT_PARAMETERS not in self._parameters_dict:
            raise KeyError("No default ParametersRecord.")
        return self._parameters_dict[DEFAULT_PARAMETERS]

    @default_parameters.setter
    def default_parameters(self, value: ParametersRecord) -> None:
        """Set default_parameters."""
        self._parameters_dict[DEFAULT_PARAMETERS] = value

    @property
    def default_metrics(self) -> MetricsRecord:
        """Default MetricsRecord."""
        if DEFAULT_METRICS not in self._metrics_dict:
            raise KeyError("No default MetricsRecord.")
        return self._metrics_dict[DEFAULT_METRICS]

    @default_metrics.setter
    def default_metrics(self, value: MetricsRecord) -> None:
        """Set default_metrics."""
        self._metrics_dict[DEFAULT_METRICS] = value

    @property
    def default_configs(self) -> ConfigsRecord:
        """Default ConfigsRecord."""
        if DEFAULT_CONFIGS not in self._configs_dict:
            raise KeyError("No default ConfigsRecord.")
        return self._configs_dict[DEFAULT_CONFIGS]

    @default_configs.setter
    def default_configs(self, value: ConfigsRecord) -> None:
        """Set default_configs."""
        self._configs_dict[DEFAULT_CONFIGS] = value
