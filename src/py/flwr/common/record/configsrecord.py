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


from typing import Dict, Optional, get_args

from flwr.common.typing import ConfigsRecordValues, ConfigsScalar

from .typeddict import TypedDict


def _check_key(key: str) -> None:
    """Check if key is of expected type."""
    if not isinstance(key, str):
        raise TypeError(f"Key must be of type `str` but `{type(key)}` was passed.")


def _check_value(value: ConfigsRecordValues) -> None:
    def is_valid(__v: ConfigsScalar) -> None:
        """Check if value is of expected type."""
        if not isinstance(__v, get_args(ConfigsScalar)):
            raise TypeError(
                "Not all values are of valid type."
                f" Expected `{ConfigsRecordValues}` but `{type(__v)}` was passed."
            )

    if isinstance(value, list):
        # If your lists are large (e.g. 1M+ elements) this will be slow
        # 1s to check 10M element list on a M2 Pro
        # In such settings, you'd be better of treating such config as
        # an array and pass it to a ParametersRecord.
        # Empty lists are valid
        if len(value) > 0:
            is_valid(value[0])
            # all elements in the list must be of the same valid type
            # this is needed for protobuf
            value_type = type(value[0])
            if not all(isinstance(v, value_type) for v in value):
                raise TypeError(
                    "All values in a list must be of the same valid type. "
                    f"One of {ConfigsScalar}."
                )
    else:
        is_valid(value)


class ConfigsRecord(TypedDict[str, ConfigsRecordValues]):
    """Configs record."""

    def __init__(
        self,
        configs_dict: Optional[Dict[str, ConfigsRecordValues]] = None,
        keep_input: bool = True,
    ) -> None:
        """Construct a ConfigsRecord object.

        Parameters
        ----------
        configs_dict : Optional[Dict[str, ConfigsRecordValues]]
            A dictionary that stores basic types (i.e. `str`, `int`, `float`, `bytes` as
            defined in `ConfigsScalar`) and lists of such types (see
            `ConfigsScalarList`).
        keep_input : bool (default: True)
            A boolean indicating whether config passed should be deleted from the input
            dictionary immediately after adding them to the record. When set
            to True, the data is duplicated in memory. If memory is a concern, set
            it to False.
        """
        super().__init__(_check_key, _check_value)
        if configs_dict:
            for k in list(configs_dict.keys()):
                self[k] = configs_dict[k]
                if not keep_input:
                    del configs_dict[k]
