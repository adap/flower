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


from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from typing import Any, cast, overload

import numpy as np

from ..constant import SType
from ..typing import NDArray
from .typeddict import TypedDict


def _raise_array_init_error() -> None:
    raise TypeError(
        f"Invalid arguments for {Array.__qualname__}. Expected either a "
        "NumPy ndarray, or explicit dtype/shape/stype/data values."
    )


@dataclass
class Array:
    """Array type.

    A dataclass containing serialized data from an array-like or tensor-like object
    along with metadata about it. The class can be initialized in one of two ways:

    1. By specifying explicit values for `dtype`, `shape`, `stype`, and `data`.
    2. By providing a NumPy ndarray (via the `ndarray` argument).

    In scenario (2), the `dtype`, `shape`, `stype`, and `data` are automatically
    derived from the input. In scenario (1), these fields must be specified manually.

    Parameters
    ----------
    dtype : Optional[str] (default: None)
        A string representing the data type of the serialized object (e.g. `"float32"`).
        Only required if you are not passing in a ndarray.

    shape : Optional[list[int]] (default: None)
        A list representing the shape of the unserialized array-like object. Only
        required if you are not passing in a ndarray.

    stype : Optional[str] (default: None)
        A string indicating the serialization mechanism used to generate the bytes in
        `data` from an array-like or tensor-like object. Only required if you are not
        passing in a ndarray.

    data : Optional[bytes] (default: None)
        A buffer of bytes containing the data. Only required if you are not passing in
        a ndarray.

    ndarray : Optional[NDArray] (default: None)
        A NumPy ndarray. If provided, the `dtype`, `shape`, `stype`, and `data`
        fields are derived automatically from it.

    Examples
    --------
    Initializing by specifying all fields directly:

    >>> arr1 = Array(
    >>>     dtype="float32",
    >>>     shape=[3, 3],
    >>>     stype="numpy.ndarray",
    >>>     data=b"serialized_data...",
    >>> )

    Initializing with a NumPy ndarray:

    >>> import numpy as np
    >>> arr2 = Array(np.random.randn(3, 3))
    """

    dtype: str
    shape: list[int]
    stype: str
    data: bytes

    @overload
    def __init__(  # noqa: E704
        self, dtype: str, shape: list[int], stype: str, data: bytes
    ) -> None: ...

    @overload
    def __init__(self, ndarray: NDArray) -> None: ...  # noqa: E704

    def __init__(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        *args: Any,
        dtype: str | None = None,
        shape: list[int] | None = None,
        stype: str | None = None,
        data: bytes | None = None,
        ndarray: NDArray | None = None,
    ) -> None:
        # Determine the initialization method and validate input arguments.
        # Support two initialization formats:
        # 1. Array(dtype: str, shape: list[int], stype: str, data: bytes)
        # 2. Array(ndarray: NDArray)

        # Initialize all arguments
        # If more than 4 positional arguments are provided, raise an error.
        if len(args) > 4:
            _raise_array_init_error()
        all_args = [None] * 4
        for i, arg in enumerate(args):
            all_args[i] = arg
        init_method: str | None = None  # Track which init method is being used

        # Try to assign a value to all_args[index] if it's not already set.
        # If an initialization method is provided, update init_method.
        def _try_set_arg(index: int, arg: Any, method: str) -> None:
            # Skip if arg is None
            if arg is None:
                return
            # Raise an error if all_args[index] is already set
            if all_args[index] is not None:
                _raise_array_init_error()
            # Raise an error if a different initialization method is already set
            nonlocal init_method
            if init_method is not None and init_method != method:
                _raise_array_init_error()
            # Set init_method and all_args[index]
            if init_method is None:
                init_method = method
            all_args[index] = arg

        # Try to set keyword arguments in all_args
        _try_set_arg(0, dtype, "direct")
        _try_set_arg(1, shape, "direct")
        _try_set_arg(2, stype, "direct")
        _try_set_arg(3, data, "direct")
        _try_set_arg(0, ndarray, "ndarray")

        # Check if all arguments are correctly set
        all_args = [arg for arg in all_args if arg is not None]

        # Handle direct field initialization
        if not init_method or init_method == "direct":
            if (
                len(all_args) == 4  # pylint: disable=too-many-boolean-expressions
                and isinstance(all_args[0], str)
                and isinstance(all_args[1], list)
                and all(isinstance(i, int) for i in all_args[1])
                and isinstance(all_args[2], str)
                and isinstance(all_args[3], bytes)
            ):
                self.dtype, self.shape, self.stype, self.data = all_args
                return

        # Handle NumPy array
        if not init_method or init_method == "ndarray":
            if len(all_args) == 1 and isinstance(all_args[0], np.ndarray):
                self.__dict__.update(self.from_numpy_ndarray(all_args[0]).__dict__)
                return

        _raise_array_init_error()

    @classmethod
    def from_numpy_ndarray(cls, ndarray: NDArray) -> Array:
        """Create Array from NumPy ndarray."""
        assert isinstance(
            ndarray, np.ndarray
        ), f"Expected NumPy ndarray, got {type(ndarray)}"
        buffer = BytesIO()
        # WARNING: NEVER set allow_pickle to true.
        # Reason: loading pickled data can execute arbitrary code
        # Source: https://numpy.org/doc/stable/reference/generated/numpy.save.html
        np.save(buffer, ndarray, allow_pickle=False)
        data = buffer.getvalue()
        return Array(
            dtype=str(ndarray.dtype),
            shape=list(ndarray.shape),
            stype=SType.NUMPY,
            data=data,
        )

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
    >>> arr = Array(arr_np)
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
        array_dict: OrderedDict[str, Array] | None = None,
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
