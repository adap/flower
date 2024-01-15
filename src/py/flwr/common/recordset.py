# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
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
from typing import Any, Dict, List, Tuple

from .typing import Value


def _check(allowed_types: Any, element: Any) -> Tuple[bool, str]:
    """Check if passed element is of allowed type."""
    msg = ""
    check = isinstance(element, allowed_types)
    if not check:
        msg = f"It must be of type `{allowed_types}` but got `{type(element)}`"

    return check, msg


class Tensor:
    """Tensor type."""

    def __init__(self) -> None:
        self._data: List[bytes]
        self._shape: List[int]
        self._dtype: str  # tbd
        self.ref: str  # future functionality


class ParameterRecord(Dict[str, Tensor]):
    """Parameter record."""

    def __setitem__(self, key: str, value: Tensor) -> None:
        """Set item after key and value checks."""
        check, mssg = _check(str, key)
        if not check:
            raise TypeError(f"Key `{key}` is of invalid key type. {mssg}")
        check, mssg = _check(Tensor, value)
        if not check:
            raise TypeError(f"Value for key `{key}` is of invalid value type. {mssg}")
        super().__setitem__(key, value)


class MetricsRecord(Dict[str, Value]):
    """Metrics record."""

    def __setitem__(self, key: str, value: Value) -> None:
        """Set item after key and value checks."""
        check, mssg = _check(str, key)
        if not check:
            raise TypeError(f"Key `{key}` is of invalid key type. {mssg}")
        check, mssg = _check(Value, value)
        if not check:
            raise TypeError(f"Value for key `{key}` is of invalid value type. {mssg}")
        super().__setitem__(key, value)


class ConfigsRecord(Dict[str, Value]):
    """Config record."""

    def __setitem__(self, key: str, value: Value) -> None:
        """Set item after key and value checks."""
        check, mssg = _check(str, key)
        if not check:
            raise TypeError(f"Key `{key}` is of invalid key type. {mssg}")
        check, mssg = _check(Value, value)
        if not check:
            raise TypeError(f"Value for key `{key}` is of invalid value type. {mssg}")
        super().__setitem__(key, value)


@dataclass
class RecordSet:
    """Definition of RecordSet."""

    parameters: Dict[str, ParameterRecord]
    metrics: Dict[str, MetricsRecord]
    configs: Dict[str, ConfigsRecord]

    def set_parameters(self, name: str, record: ParameterRecord) -> None:
        """Add a ParameterRecord."""
        self.parameters[name] = record

    def get_parameters(self, name: str) -> ParameterRecord:
        """Get a ParameterRecord."""
        return self.parameters[name]

    def set_metrics(self, name: str, record: MetricsRecord) -> None:
        """Add a MetricsRecord."""
        self.metrics[name] = record

    def get_metrics(self, name: str) -> MetricsRecord:
        """Get a MetricsRecord."""
        return self.metrics[name]

    def set_configs(self, name: str, record: ConfigsRecord) -> None:
        """Add a ConfigsRecord."""
        self.configs[name] = record

    def get_configs(self, name: str) -> ConfigsRecord:
        """Get a ConfigsRecord."""
        return self.configs[name]
