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
from typing import Any

from .typing import Parameters, Value


def _check(allowed_types: Any, element: Any) -> [bool, str]:
    """Check if passed element is of allowed type."""
    msg = ""
    check = isinstance(element, allowed_types)
    if not check:
        msg = f"It must be of type `{allowed_types}` but got `{type(element)}`"

    return check, msg


class TypeCheckedDict(dict):
    """A dictionary with key and values checks for type validity."""

    def __init__(self):
        pass

    def __setitem__(self, key, value):
        """Set item after key and value checks."""
        check, mssg = _check(self._key_types, key)
        if not check:
            raise TypeError(f"Key `{key}` is of invalid key type. {mssg}")
        check, mssg = _check(self._value_types, value)
        if not check:
            raise TypeError(
                f"Value `{value}` for key `{key}` is of invalid value type. {mssg}"
            )
        super().__setitem__(key, value)


class TypeCheckedParametersDict(TypeCheckedDict):
    """A TypeCheckedDict for parameters."""

    def __init__(self):
        self._key_types = str
        self._value_types = Parameters


class TypeCheckedMetricsDict(TypeCheckedDict):
    """A TypeCheckedDict for metrics."""

    def __init__(self):
        self._key_types = (str, int)
        self._value_types = Value


class TypeCheckedConfigsDict(TypeCheckedDict):
    """A TypeCheckedDict for configs."""

    def __init__(self):
        self._key_types = str
        self._value_types = Value


@dataclass
class RecordSet:
    """Definition of RecordSet."""

    parameters: TypeCheckedParametersDict
    metrics: TypeCheckedMetricsDict
    configs: TypeCheckedConfigsDict
