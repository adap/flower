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

import sys
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from logging import WARN
from typing import TYPE_CHECKING, Any, cast, overload

import numpy as np

from ..constant import SType
from ..logger import log
from ..typing import NDArray
from .typeddict import TypedDict

if TYPE_CHECKING:
    import tensorflow as tf
    import torch


def _raise_array_init_error() -> None:
    raise TypeError(
        f"Invalid arguments for {Array.__qualname__}. Expected either a "
        "PyTorch tensor, TensorFlow tensor, NumPy ndarray, or explicit"
        " dtype/shape/stype/data values."
    )


def _raise_parameters_record_init_error() -> None:
    raise TypeError(
        f"Invalid arguments for {ParametersRecord.__qualname__}. Expected either "
        "a list of NumPy ndarrays, a PyTorch state_dict, TensorFlow weights, "
        "or a dictionary of Arrays. The `keep_input` argument is only supported when "
        "passing a dictionary of Arrays, and it must be specified as a keyword "
        "argument."
    )


@dataclass
class Array:
    """Array type.

    A dataclass containing serialized data from an array-like or tensor-like object
    along with metadata about it. The class can be initialized in one of four ways:

    1. By providing a PyTorch tensor (via the `torch_tensor` argument).
    2. By providing a TensorFlow tensor (via the `tf_tensor` argument).
    3. By providing a NumPy ndarray (via the `ndarray` argument).
    4. By specifying explicit values for `dtype`, `shape`, `stype`, and `data`.

    In scenarios (1)-(3), the `dtype`, `shape`, `stype`, and `data` are automatically
    derived from the provided tensor or ndarray. In scenario (4), these fields must be
    specified manually.

    Parameters
    ----------
    torch_tensor : Optional[torch.Tensor] (default: None)
        A PyTorch tensor. If provided, it will be **detached and moved to CPU**
        before conversion, and the `dtype`, `shape`, `stype`, and `data` fields
        will be derived automatically from it.

    tf_tensor : Optional[tf.Tensor] (default: None)
        A TensorFlow tensor. If provided, the `dtype`, `shape`, `stype`, and `data`
        fields are derived automatically from it.

    ndarray : Optional[NDArray] (default: None)
        A NumPy ndarray. If provided, the `dtype`, `shape`, `stype`, and `data`
        fields are derived automatically from it.

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

    Examples
    --------
    Initializing with a NumPy ndarray:

    >>> import numpy as np
    >>> arr1 = Array(np.random.randn(3, 3))

    Initializing with a PyTorch tensor:

    >>> import torch
    >>> arr2 = Array(torch.randn(3, 3))

    Initializing with a TensorFlow tensor:

    >>> import tensorflow as tf
    >>> arr3 = Array(tf.random.normal([3, 3]))

    Initializing by specifying all fields directly:

    >>> arr2 = Array(
    >>>     dtype="float32",
    >>>     shape=[3, 3],
    >>>     stype="numpy.ndarray",
    >>>     data=b"serialized_data...",
    >>> )
    """

    dtype: str
    shape: list[int]
    stype: str
    data: bytes

    @overload
    def __init__(self, torch_tensor: torch.Tensor) -> None: ...  # noqa: E704

    @overload
    def __init__(self, tf_tensor: tf.Tensor) -> None: ...  # noqa: E704

    @overload
    def __init__(self, ndarray: NDArray) -> None: ...  # noqa: E704

    @overload
    def __init__(  # noqa: E704
        self, dtype: str, shape: list[int], stype: str, data: bytes
    ) -> None: ...

    def __init__(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        *args: Any,
        torch_tensor: torch.Tensor | None = None,
        tf_tensor: tf.Tensor | None = None,
        ndarray: NDArray | None = None,
        dtype: str | None = None,
        shape: list[int] | None = None,
        stype: str | None = None,
        data: bytes | None = None,
    ) -> None:
        # Workaround to support multiple initialization signatures.
        # This method validates and assigns the correct arguments,
        # including keyword arguments such as dtype and shape.
        # Supported initialization formats:
        # 1. Array(dtype: str, shape: list[int], stype: str, data: bytes)
        # 2. Array(ndarray: NDArray)
        # 3. Array(torch_tensor: torch.Tensor)
        # 4. Array(tf_tensor: tf.Tensor)

        # Init all arguments
        # If more than 4 positional arguments are provided, raise an error.
        if len(args) > 4:
            _raise_array_init_error()
        all_args = [None] * 4
        for i, arg in enumerate(args):
            all_args[i] = arg

        def _try_set_arg(index: int, arg: Any) -> None:
            if arg is None:
                return
            if all_args[index] is not None:
                _raise_array_init_error()
            all_args[index] = arg

        # Try to set keyword arguments in all_args
        _try_set_arg(0, torch_tensor)
        _try_set_arg(0, tf_tensor)
        _try_set_arg(0, ndarray)
        _try_set_arg(0, dtype)
        _try_set_arg(1, shape)
        _try_set_arg(2, stype)
        _try_set_arg(3, data)

        # Check if all arguments are correctly set
        all_args = [arg for arg in all_args if arg is not None]
        if len(all_args) not in [1, 4]:
            _raise_array_init_error()

        # Handle PyTorch tensor
        if "torch" in sys.modules and isinstance(
            all_args[0], sys.modules["torch"].Tensor
        ):
            self.__dict__.update(self.from_torch_tensor(all_args[0]).__dict__)
            return

        # Handle TensorFlow tensor
        if "tensorflow" in sys.modules and isinstance(
            all_args[0], sys.modules["tensorflow"].Tensor
        ):
            self.__dict__.update(self.from_tf_tensor(all_args[0]).__dict__)
            return

        # Handle NumPy array
        if isinstance(all_args[0], np.ndarray):
            self.__dict__.update(self.from_numpy_ndarray(all_args[0]).__dict__)
            return

        # Handle direct field initialization
        if (
            isinstance(all_args[0], str)
            and isinstance(all_args[1], list)
            and all(isinstance(i, int) for i in all_args[1])
            and isinstance(all_args[2], str)
            and isinstance(all_args[3], bytes)
        ):
            self.dtype, self.shape, self.stype, self.data = all_args
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

    @classmethod
    def from_torch_tensor(cls, tensor: torch.Tensor) -> Array:
        """Create Array from PyTorch tensor."""
        if not (torch := sys.modules.get("torch")):
            raise RuntimeError(
                f"PyTorch is required to use {cls.from_torch_tensor.__name__}"
            )

        assert isinstance(
            tensor, torch.Tensor
        ), f"Expected PyTorch Tensor, got {type(tensor)}"
        return cls.from_numpy_ndarray(tensor.detach().cpu().numpy())

    @classmethod
    def from_tf_tensor(cls, tensor: tf.Tensor) -> Array:
        """Create Array from TensorFlow tensor."""
        if not (tf := sys.modules.get("tensorflow")):
            raise RuntimeError(
                f"TensorFlow is required to use {cls.from_tf_tensor.__name__}"
            )

        assert isinstance(
            tensor, tf.Tensor
        ), f"Expected TensorFlow Tensor, got {type(tensor)}"
        return cls.from_numpy_ndarray(tensor.numpy())

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

    A typed dictionary (string to :class:`Array`) that can store named parameters
    (serialized tensors). Internally, this behaves similarly to an
    `OrderedDict[str, Array]`. A `ParametersRecord` can be viewed as an
    equivalent to PyTorch's state_dict, but it holds arrays in serialized form.

    This object is one of the record types supported by :class:`RecordSet` and can
    therefore be stored in a :class:`Message` or a :class:`Context`.

    This class can be instantiated in multiple ways:

    1. By providing nothing (empty container).
    2. By providing a PyTorch state_dict (via the `state_dict` argument).
    3. By providing TensorFlow model.get_weights() (via the `tf_weights` argument).
    4. By providing a list of NumPy ndarrays (via the `numpy_ndarrays` argument).
    5. By providing a dictionary of Arrays (via the `array_dict` argument).

    The `keep_input` argument is only supported when passing a dictionary of Arrays.

    Parameters
    ----------
    numpy_ndarrays : Optional[list[NDArray]] (default: None)
        A list of NumPy arrays. Each array will be automatically converted
        into an :class:`Array` and stored in this record with generated keys.

    state_dict : Optional[OrderedDict[str, torch.Tensor]] (default: None)
        A PyTorch state_dict (str keys to torch.Tensor values). Each
        tensor will be converted into an :class:`Array` and stored in this record.

    tf_weights : Optional[list[NDArray]] (default: None)
        TensorFlow model weights (which are typically NumPy ndarrays when
        accessed via `model.get_weights()`). Each weight will be converted into
        an :class:`Array` and stored in this record.

    array_dict : Optional[OrderedDict[str, Array]] (default: None)
        An existing dictionary containing named :class:`Array` objects. If
        provided, these entries will be used directly to populate the record.

    keep_input : Optional[bool] (default: None)
        If `False` (default), entries in `array_dict` are removed after being added
        to this record to free up memory. If `True`, the original `array_dict`
        remains unchanged, preserving the data in both places at the cost of increased
        memory usage.

    Examples
    --------
    Initializing an empty ParametersRecord:

    >>> p_record = ParametersRecord()

    Initializing with a PyTorch model state_dict:

    >>> import torch.nn as nn
    >>>
    >>> model = nn.Linear(10, 5)
    >>> p_record = ParametersRecord(model.state_dict())

    Initializing with a TensorFlow model weights (a list of NumPy arrays):

    >>> import tensorflow as tf
    >>>
    >>> model = tf.keras.Sequential([tf.keras.layers.Dense(5, input_shape=(10,))])
    >>> p_record = ParametersRecord(model.get_weights())

    Initializing with a list of NumPy arrays:

    >>> import numpy as np
    >>>
    >>> arr1 = np.random.randn(3, 3)
    >>> arr2 = np.random.randn(2, 2)
    >>> p_record = ParametersRecord(numpy_ndarrays=[arr1, arr2])
    """

    @overload
    def __init__(self) -> None: ...  # noqa: E704

    @overload
    def __init__(self, numpy_ndarrays: list[NDArray]) -> None: ...  # noqa: E704

    @overload
    def __init__(  # noqa: E704
        self, state_dict: OrderedDict[str, torch.Tensor]
    ) -> None: ...

    @overload
    def __init__(self, tf_weights: list[NDArray]) -> None: ...  # noqa: E704

    @overload
    def __init__(  # noqa: E704
        self, array_dict: OrderedDict[str, Array], *, keep_input: bool
    ) -> None: ...

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *args: Any,
        numpy_ndarrays: list[NDArray] | None = None,
        state_dict: OrderedDict[str, torch.Tensor] | None = None,
        tf_weights: list[NDArray] | None = None,
        array_dict: OrderedDict[str, Array] | None = None,
        keep_input: bool | None = None,
    ) -> None:
        super().__init__(_check_key, _check_value)

        # Workaround to support multiple initialization signatures.
        # This method validates and assigns the correct arguments,
        # including keyword arguments such as numpy_ndarrays, state_dict,
        # tf_weights, and array_dict.
        # Supported initialization formats:
        # 1. ParametersRecord(numpy_ndarrays: list[NDArray])
        # 2. ParametersRecord(state_dict: dict[str, torch.Tensor])
        # 3. ParametersRecord(tf_weights: list[NDArray])
        # 4. ParametersRecord(array_dict: OrderedDict[str, Array], keep_input: bool)

        # Init the argument
        if len(args) > 1:
            _raise_parameters_record_init_error()

        arg = args[0] if args else None

        def _try_set_arg(_arg_to_set: Any) -> None:
            nonlocal arg
            if _arg_to_set is None:
                return
            if arg is not None:
                _raise_parameters_record_init_error()
            arg = _arg_to_set

        # Try to set keyword arguments
        _try_set_arg(numpy_ndarrays)
        _try_set_arg(state_dict)
        _try_set_arg(tf_weights)
        _try_set_arg(array_dict)

        # If no arguments are provided, return and keep self empty
        if arg is None:
            if keep_input is not None:
                log(WARN, "`keep_input` will be ignored. No parameters were provided.")
            return

        # Handle dictionary of Arrays
        if isinstance(arg, dict) and all(isinstance(v, Array) for v in arg.values()):
            array_dict = cast(OrderedDict[str, Array], arg)
            if keep_input is None:
                keep_input = False

            for k in list(array_dict.keys()):
                self[k] = array_dict[k]
                if not keep_input:
                    del array_dict[k]
            return

        # Check if keep_input is set
        if keep_input is not None:
            log(
                WARN,
                "`keep_input` will be ignored. It is only supported when "
                "passing a dictionary of Arrays.",
            )

        # Handle NumPy ndarrays and TensorFlow weights
        # pylint: disable-next=not-an-iterable
        if isinstance(arg, list) and all(isinstance(v, np.ndarray) for v in arg):
            numpy_ndarrays = cast(list[NDArray], arg)
            # Skip updating if arg is empty
            if numpy_ndarrays:
                self.__dict__.update(self.from_numpy_ndarrays(numpy_ndarrays).__dict__)
            return

        # Handle PyTorch state_dict
        if (
            (torch := sys.modules.get("torch")) is not None
            and isinstance(arg, dict)
            and all(isinstance(k, str) for k in arg)  # pylint: disable=not-an-iterable
            and all(isinstance(v, torch.Tensor) for v in arg.values())
        ):
            state_dict = cast(OrderedDict[str, torch.Tensor], arg)  # type: ignore
            # Skip updating if arg is empty
            if state_dict:
                self.__dict__.update(self.from_state_dict(state_dict).__dict__)
            return

        _raise_parameters_record_init_error()

    @classmethod
    def from_numpy_ndarrays(
        cls,
        ndarrays: list[NDArray],
    ) -> ParametersRecord:
        """Create ParametersRecord from a list of NumPy arrays."""
        record = ParametersRecord()
        for i, arr in enumerate(ndarrays):
            record[str(i)] = Array.from_numpy_ndarray(arr)
        return record

    @classmethod
    def from_state_dict(
        cls,
        state_dict: OrderedDict[str, torch.Tensor],
    ) -> ParametersRecord:
        """Create ParametersRecord from PyTorch state_dict."""
        if "torch" not in sys.modules:
            raise RuntimeError(
                f"PyTorch is required to use {cls.from_state_dict.__name__}"
            )

        record = ParametersRecord()
        for k, v in state_dict.items():
            record[k] = Array.from_numpy_ndarray(v.detach().cpu().numpy())
        return record

    @classmethod
    def from_tf_weights(
        cls,
        tf_weights: list[NDArray],
    ) -> ParametersRecord:
        """Create ParametersRecord from TensorFlow weights."""
        return cls.from_numpy_ndarrays(tf_weights)

    def to_numpy_ndarrays(self) -> list[NDArray]:
        """Return the ParametersRecord as a list of NumPy arrays."""
        return [v.numpy() for v in self.values()]

    def to_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        """Return the ParametersRecord as a PyTorch state_dict."""
        if not (torch := sys.modules.get("torch")):
            raise RuntimeError(
                f"PyTorch is required to use {self.to_state_dict.__name__}"
            )

        state_dict = OrderedDict()
        for k, v in self.items():
            state_dict[k] = torch.from_numpy(v.numpy())
        return state_dict

    def to_tf_weights(self) -> list[NDArray]:
        """Return the ParametersRecord as a list of TensorFlow weights."""
        return self.to_numpy_ndarrays()

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
