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
"""ParametersRecord and Array."""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Array:
    """Array type."""

    dtype: str
    shape: List[int]
    stype: str
    data: bytes


@dataclass
class ParametersRecord:
    """Parameters record."""

    data: OrderedDict[str, Array] = field(default_factory=OrderedDict[str, Array])

    def add_parameters(self, array_dict: Dict[str, Array]) -> None:
        """Add parameters to record.

        This is not implemented as a constructor so we can cleanly create and empty the
        ParametersRecord object.
        """
        if any(not isinstance(k, str) for k in array_dict.keys()):
            raise TypeError(f"Not all keys are of valide type. Expected {str}")
        if any(not isinstance(v, Array) for v in array_dict.values()):
            raise TypeError(f"Not all values are of valide type. Expected {Array}")

        # Add entries to dataclass without duplicating memory
        for key in list(array_dict.keys()):
            self.data[key] = array_dict[key]
            del array_dict[key]
