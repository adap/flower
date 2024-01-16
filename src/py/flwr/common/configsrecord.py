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
"""ConfigsRecord."""

from dataclasses import dataclass, field
from typing import Dict, Union, get_args

from .typing import Scalar, ScalarList

ConfigKeys = Union[str, int]


@dataclass
class ConfigsRecord:
    """Configs record."""

    data: Dict[ConfigKeys, Union[Scalar, ScalarList]] = field(default_factory=dict)

    def add_configs(
        self, configs_dict: Dict[ConfigKeys, Union[Scalar, ScalarList]]
    ) -> None:
        """Add configs to record.

        This not implemented as a constructor so we can cleanly create and empyt
        ConfigsRecord object.
        """
        if any(not isinstance(k, get_args(ConfigKeys)) for k in configs_dict.keys()):
            raise TypeError(f"Not all keys are of valide type. Expected {ConfigKeys}")

        def is_valid(value: Scalar) -> None:
            """Check if value is of expected type."""
            if not isinstance(value, get_args(Scalar)):
                raise TypeError(
                    "Not all values are of valide type."
                    f" Expected {Union[Scalar, ScalarList]}"
                )

        # Check types of values
        # Split between those values that are list and those that aren't
        # then process in the same way
        for value in configs_dict.values():
            if isinstance(value, list):
                # If your lists are large (e.g. 1M+ elements) this will be slow
                # 1s to check 10M element list on a M2 Pro
                # In such settings, you'd be better of treating such metric as
                # an array and pass it to a ParametersRecord.
                for list_value in value:
                    is_valid(list_value)
            else:
                is_valid(value)

        # Add entries to dataclass without duplicating memory
        for key in list(configs_dict.keys()):
            self.data[key] = configs_dict[key]
            del configs_dict[key]
