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


from dataclasses import dataclass, field
from typing import Dict, Iterable, Union

from .configsrecord import ConfigsRecord
from .metricsrecord import MetricsRecord
from .parametersrecord import ParametersRecord


@dataclass
class RecordSet:
    """Enhanced RecordSet with a unified and Pythonic interface."""

    _parameters: Dict[str, ParametersRecord] = field(default_factory=dict)
    _metrics: Dict[str, MetricsRecord] = field(default_factory=dict)
    _configs: Dict[str, ConfigsRecord] = field(default_factory=dict)

    def __getitem__(
        self, key: str
    ) -> Union[ParametersRecord, MetricsRecord, ConfigsRecord]:
        """."""
        if key in self._parameters:
            return self._parameters[key]
        elif key in self._metrics:
            return self._metrics[key]
        elif key in self._configs:
            return self._configs[key]
        raise KeyError(f"Invalid key: {key}")

    def __setitem__(
        self, key: str, value: Union[ParametersRecord, MetricsRecord, ConfigsRecord]
    ) -> None:
        """."""
        if isinstance(value, ParametersRecord):
            self._parameters[key] = value
        elif isinstance(value, MetricsRecord):
            self._metrics[key] = value
        elif isinstance(value, ConfigsRecord):
            self._configs[key] = value
        else:
            raise ValueError(f"Invalid value type: {type(value)}")

    def __delitem__(self, key: str) -> None:
        """."""
        if key in self._parameters:
            del self._parameters[key]
        elif key in self._metrics:
            del self._metrics[key]
        elif key in self._configs:
            del self._configs[key]
        else:
            raise KeyError(f"Invalid key: {key}")

    def __str__(self) -> str:
        """."""
        return (
            f"RecordSet(parameters={self._parameters}, "
            f"metrics={self._metrics}, configs={self._configs})"
        )

    def __repr__(self) -> str:
        """."""
        return self.__str__()

    def paramsrecord_keys(self) -> Iterable[str]:
        """Retrieve the keys for each stored `ParametersRecord`."""
        return self._parameters.keys()

    def metricsrecord_keys(self) -> Iterable[str]:
        """Retrieve the keys for each stored `MetricsRecord`."""
        return self._metrics.keys()

    def configsrecord_keys(self) -> Iterable[str]:
        """Retrieve the keys for each stored `ConfigsRecord`."""
        return self._configs.keys()
