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
"""ParametersRecord and Tensor."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Array:
    """Array type."""

    data: bytes
    dtype: str
    stype: str
    shape: List[int]
    ref: str = ""  # future functionality


@dataclass
class ParametersRecord:
    """Parameters record."""

    data: Dict[str, Array] = field(default_factory=dict)

    def add_parameters(self, tensor_dict: Dict[str, Array]) -> None:
        """Add parameters to record.

        This not implemented as a constructor so we can cleanly create and empyt
        ParametersRecord object.
        """
        if any(not isinstance(k, str) for k in tensor_dict.keys()):
            raise TypeError(f"Not all keys are of valide type. Expected {str}")
        if any(not isinstance(v, Array) for v in tensor_dict.values()):
            raise TypeError(f"Not all values are of valide type. Expected {Array}")

        # Add entries to dataclass without duplicating memory
        for key in list(tensor_dict.keys()):
            self.data[key] = tensor_dict[key]
            del tensor_dict[key]
