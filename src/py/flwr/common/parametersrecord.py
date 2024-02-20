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


from dataclasses import dataclass
from typing import List, Optional, OrderedDict

from .typeddict import TypedDict


@dataclass
class Array:
    """Array type.

    A dataclass containing serialized data from an array-like or tensor-like object
    along with some metadata about it.

    Parameters
    ----------
    dtype : str
        A string representing the data type of the serialised object (e.g. `np.float32`)

    shape : List[int]
        A list representing the shape of the unserialized array-like object. This is
        used to deserialize the data (depending on the serialization method) or simply
        as a metadata field.

    stype : str
        A string indicating the type of serialisation mechanism used to generate the
        bytes in `data` from an array-like or tensor-like object.

    data: bytes
        A buffer of bytes containing the data.
    """

    dtype: str
    shape: List[int]
    stype: str
    data: bytes


def _check_key(key: str) -> None:
    """Check if key is of expected type."""
    if not isinstance(key, str):
        raise TypeError(f"Key must be of type `str` but `{type(key)}` was passed.")


def _check_value(value: Array) -> None:
    if not isinstance(value, Array):
        raise TypeError(
            f"Value must be of type `{Array}` but `{type(value)}` was passed."
        )


@dataclass
class ParametersRecord(TypedDict[str, Array]):
    """Parameters record.

    A dataclass storing named Arrays in order. This means that it holds entries as an
    OrderedDict[str, Array]. ParametersRecord objects can be viewed as an equivalent to
    PyTorch's state_dict, but holding serialised tensors instead.
    """

    def __init__(
        self,
        array_dict: Optional[OrderedDict[str, Array]] = None,
        keep_input: bool = False,
    ) -> None:
        """Construct a ParametersRecord object.

        Parameters
        ----------
        array_dict : Optional[OrderedDict[str, Array]]
            A dictionary that stores serialized array-like or tensor-like objects.
        keep_input : bool (default: False)
            A boolean indicating whether parameters should be deleted from the input
            dictionary immediately after adding them to the record. If False, the
            dictionary passed to `set_parameters()` will be empty once exiting from that
            function. This is the desired behaviour when working with very large
            models/tensors/arrays. However, if you plan to continue working with your
            parameters after adding it to the record, set this flag to True. When set
            to True, the data is duplicated in memory.
        """
        super().__init__(_check_key, _check_value)
        if array_dict:
            for k in list(array_dict.keys()):
                self[k] = array_dict[k]
                if not keep_input:
                    del array_dict[k]
