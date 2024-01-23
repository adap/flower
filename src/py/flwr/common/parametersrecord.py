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


from dataclasses import dataclass, field
from typing import List, Optional, OrderedDict


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


@dataclass
class ParametersRecord:
    """Parameters record.

    A dataclass storing named Arrays in order. This means that it holds entries as an
    OrderedDict[str, Array]. ParametersRecord objects can be viewed as an equivalent to
    PyTorch's state_dict, but holding serialised tensors instead.
    """

    data: OrderedDict[str, Array] = field(default_factory=OrderedDict[str, Array])

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
        self.data = OrderedDict()
        if array_dict:
            self.set_parameters(array_dict, keep_input=keep_input)

    def set_parameters(
        self, array_dict: OrderedDict[str, Array], keep_input: bool = False
    ) -> None:
        """Add parameters to record.

        Parameters
        ----------
        array_dict : OrderedDict[str, Array]
            A dictionary that stores serialized array-like or tensor-like objects.
        keep_input : bool (default: False)
            A boolean indicating whether parameters should be deleted from the input
            dictionary immediately after adding them to the record.
        """
        if any(not isinstance(k, str) for k in array_dict.keys()):
            raise TypeError(f"Not all keys are of valid type. Expected {str}")
        if any(not isinstance(v, Array) for v in array_dict.values()):
            raise TypeError(f"Not all values are of valid type. Expected {Array}")

        if keep_input:
            # Copy
            self.data = OrderedDict(array_dict)
        else:
            # Add entries to dataclass without duplicating memory
            for key in list(array_dict.keys()):
                self.data[key] = array_dict[key]
                del array_dict[key]

    def __getitem__(self, key: str) -> Array:
        """Retrieve an element stored in record."""
        return self.data[key]
