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
from dataclasses import dataclass
from io import BytesIO
from typing import Optional, cast

import numpy as np

from ..constant import SType
from ..typing import NDArray
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
    shape: list[int]
    stype: str
    data: bytes

    def numpy(self) -> NDArray:
        """Return the array as a NumPy array."""
        if self.stype != SType.NUMPY:
            raise TypeError(
                f"Unsupported serialization type for numpy conversion: '{self.stype}'"
            )
        bytes_io = BytesIO(self.data)
        # WARNING: NEVER set allow_pickle to true.
        # Reason: loading pickled data can execute arbitrary code
        # Source: https://numpy.org/doc/stable/reference/generated/numpy.load.html
        ndarray_deserialized = np.load(bytes_io, allow_pickle=False)
        return cast(NDArray, ndarray_deserialized)


def _check_key(key: str) -> None:
    """Check if key is of expected type."""
    if not isinstance(key, str):
        raise TypeError(f"Key must be of type `str` but `{type(key)}` was passed.")


def _check_value(value: Array) -> None:
    if not isinstance(value, Array):
        raise TypeError(
            f"Value must be of type `{Array}` but `{type(value)}` was passed."
        )


class ParametersRecord(TypedDict[str, Array]):
    r"""Parameters record.

    A dataclass storing named Arrays in order. This means that it holds entries as an
    OrderedDict[str, Array]. ParametersRecord objects can be viewed as an equivalent to
    PyTorch's state_dict, but holding serialised tensors instead. A
    :code:`ParametersRecord`  is one of the types of records that a
    `flwr.common.RecordSet <flwr.common.RecordSet.html#recordset>`_ supports and
    can therefore be used to construct :code:`common.Message` objects.

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

    Examples
    --------
    The usage of :code:`ParametersRecord` is envisioned for storing data arrays (e.g.
    parameters of a machine learning model). These first need to be serialized into
    a :code:`flwr.common.Array` data structure.

    Let's see some examples:

    >>> import numpy as np
    >>> from flwr.common import ParametersRecord
    >>> from flwr.common import array_from_numpy
    >>>
    >>> # Let's create a simple NumPy array
    >>> arr_np = np.random.randn(3, 3)
    >>>
    >>> # If we print it
    >>> array([[-1.84242409, -1.01539537, -0.46528405],
    >>>      [ 0.32991896,  0.55540414,  0.44085534],
    >>>      [-0.10758364,  1.97619858, -0.37120501]])
    >>>
    >>> # Let's create an Array out of it
    >>> arr = array_from_numpy(arr_np)
    >>>
    >>> # If we print it you'll see (note the binary data)
    >>> Array(dtype='float64', shape=[3,3], stype='numpy.ndarray', data=b'@\x99\x18...')
    >>>
    >>> # Adding it to a ParametersRecord:
    >>> p_record = ParametersRecord({"my_array": arr})

    Now that the NumPy array is embedded into a :code:`ParametersRecord` it could be
    sent if added as part of a :code:`common.Message` or it could be saved as a
    persistent state of a :code:`ClientApp` via its context. Regardless of the usecase,
    we will sooner or later want to recover the array in its original NumPy
    representation. For the example above, where the array was serialized using the
    built-in utility function, deserialization can be done as follows:

    >>> # Use the Array's built-in method
    >>> arr_np_d = arr.numpy()
    >>>
    >>> # If printed, it will show the exact same data as above:
    >>> array([[-1.84242409, -1.01539537, -0.46528405],
    >>>      [ 0.32991896,  0.55540414,  0.44085534],
    >>>      [-0.10758364,  1.97619858, -0.37120501]])

    If you need finer control on how your arrays are serialized and deserialized, you
    can construct :code:`Array` objects directly like this:

    >>> from flwr.common import Array
    >>> # Serialize your array and construct Array object
    >>> arr = Array(
    >>>         data=ndarray.tobytes(),
    >>>         dtype=str(ndarray.dtype),
    >>>         stype="",  # Could be used in a deserialization function
    >>>         shape=list(ndarray.shape),
    >>>       )
    >>>
    >>> # Then you can deserialize it like this
    >>> arr_np_d = np.frombuffer(
    >>>             buffer=array.data,
    >>>             dtype=array.dtype,
    >>>            ).reshape(array.shape)

    Note that different arrays (e.g. from PyTorch, Tensorflow) might require different
    serialization mechanism. Howerver, they often support a conversion to NumPy,
    therefore allowing to use the same or similar steps as in the example above.
    """

    def __init__(
        self,
        array_dict: Optional[OrderedDict[str, Array]] = None,
        keep_input: bool = False,
    ) -> None:
        super().__init__(_check_key, _check_value)
        if array_dict:
            for k in list(array_dict.keys()):
                self[k] = array_dict[k]
                if not keep_input:
                    del array_dict[k]

    def count_bytes(self) -> int:
        """Return number of Bytes stored in this object.

        Note that a small amount of Bytes might also be included in this counting that
        correspond to metadata of the serialized object (e.g. of NumPy array) needed for
        deseralization.
        """
        num_bytes = 0

        for k, v in self.items():
            num_bytes += len(v.data)

            # We also count the bytes footprint of the keys
            num_bytes += len(k)

        return num_bytes
