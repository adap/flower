# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Array."""


from __future__ import annotations

import sys
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Any, cast, overload

import numpy as np

from ..constant import SType
from ..typing import NDArray

if TYPE_CHECKING:
    import torch


def _raise_array_init_error() -> None:
    raise TypeError(
        f"Invalid arguments for {Array.__qualname__}. Expected either a "
        "PyTorch tensor, a NumPy ndarray, or explicit"
        " dtype/shape/stype/data values."
    )


@dataclass
class Array:
    """Array type.

    A dataclass containing serialized data from an array-like or tensor-like object
    along with metadata about it. The class can be initialized in one of three ways:

    1. By specifying explicit values for `dtype`, `shape`, `stype`, and `data`.
    2. By providing a NumPy ndarray (via the `ndarray` argument).
    3. By providing a PyTorch tensor (via the `torch_tensor` argument).

    In scenarios (2)-(3), the `dtype`, `shape`, `stype`, and `data` are automatically
    derived from the input. In scenario (1), these fields must be specified manually.

    Parameters
    ----------
    dtype : Optional[str] (default: None)
        A string representing the data type of the serialized object (e.g. `"float32"`).
        Only required if you are not passing in a ndarray or a tensor.

    shape : Optional[list[int]] (default: None)
        A list representing the shape of the unserialized array-like object. Only
        required if you are not passing in a ndarray or a tensor.

    stype : Optional[str] (default: None)
        A string indicating the serialization mechanism used to generate the bytes in
        `data` from an array-like or tensor-like object. Only required if you are not
        passing in a ndarray or a tensor.

    data : Optional[bytes] (default: None)
        A buffer of bytes containing the data. Only required if you are not passing in
        a ndarray or a tensor.

    ndarray : Optional[NDArray] (default: None)
        A NumPy ndarray. If provided, the `dtype`, `shape`, `stype`, and `data`
        fields are derived automatically from it.

    torch_tensor : Optional[torch.Tensor] (default: None)
        A PyTorch tensor. If provided, it will be **detached and moved to CPU**
        before conversion, and the `dtype`, `shape`, `stype`, and `data` fields
        will be derived automatically from it.

    Examples
    --------
    Initializing by specifying all fields directly::

        arr1 = Array(
            dtype="float32",
            shape=[3, 3],
            stype="numpy.ndarray",
            data=b"serialized_data...",
        )

    Initializing with a NumPy ndarray::

        import numpy as np
        arr2 = Array(np.random.randn(3, 3))

    Initializing with a PyTorch tensor::

        import torch
        arr3 = Array(torch.randn(3, 3))
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

    @overload
    def __init__(self, torch_tensor: torch.Tensor) -> None: ...  # noqa: E704

    def __init__(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        *args: Any,
        dtype: str | None = None,
        shape: list[int] | None = None,
        stype: str | None = None,
        data: bytes | None = None,
        ndarray: NDArray | None = None,
        torch_tensor: torch.Tensor | None = None,
    ) -> None:
        # Determine the initialization method and validate input arguments.
        # Support three initialization formats:
        # 1. Array(dtype: str, shape: list[int], stype: str, data: bytes)
        # 2. Array(ndarray: NDArray)
        # 3. Array(torch_tensor: torch.Tensor)

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
        _try_set_arg(0, torch_tensor, "torch_tensor")

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

        # Handle PyTorch tensor
        if not init_method or init_method == "torch_tensor":
            if (
                len(all_args) == 1
                and "torch" in sys.modules
                and isinstance(all_args[0], sys.modules["torch"].Tensor)
            ):
                self.__dict__.update(self.from_torch_tensor(all_args[0]).__dict__)
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
