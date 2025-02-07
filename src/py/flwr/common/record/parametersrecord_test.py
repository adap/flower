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
"""Unit tests for ParametersRecord and Array."""


import unittest
from collections import OrderedDict
from io import BytesIO
from typing import Any

import numpy as np
import pytest
from parameterized import parameterized

from flwr.common import ndarray_to_bytes

from ..constant import SType
from ..typing import NDArray
from .parametersrecord import Array, ParametersRecord


def _get_buffer_from_ndarray(array: NDArray) -> bytes:
    """Return a bytes buffer froma given NumPy array."""
    buffer = BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


class TestArray(unittest.TestCase):
    """Unit tests for Array."""

    def test_numpy_conversion_valid(self) -> None:
        """Test the numpy method with valid Array instance."""
        # Prepare
        original_array = np.array([1, 2, 3], dtype=np.float32)

        buffer = _get_buffer_from_ndarray(original_array)

        # Execute
        array_instance = Array(
            dtype=str(original_array.dtype),
            shape=list(original_array.shape),
            stype=SType.NUMPY,
            data=buffer,
        )
        converted_array = array_instance.numpy()

        # Assert
        np.testing.assert_array_equal(converted_array, original_array)

    def test_numpy_conversion_invalid(self) -> None:
        """Test the numpy method with invalid Array instance."""
        # Prepare
        array_instance = Array(
            dtype="float32",
            shape=[3],
            stype="invalid_stype",  # Non-numpy stype
            data=b"",
        )

        # Execute and assert
        with self.assertRaises(TypeError):
            array_instance.numpy()

    def test_array_from_numpy(self) -> None:
        """Test the array_from_numpy function."""
        # Prepare
        original_array = np.array([1, 2, 3], dtype=np.float32)

        # Execute
        array_instance = Array.from_numpy_ndarray(original_array)
        buffer = BytesIO(array_instance.data)
        deserialized_array = np.load(buffer, allow_pickle=False)

        # Assert
        self.assertEqual(array_instance.dtype, str(original_array.dtype))
        self.assertEqual(array_instance.shape, list(original_array.shape))
        self.assertEqual(array_instance.stype, SType.NUMPY)
        np.testing.assert_array_equal(deserialized_array, original_array)

    @parameterized.expand(  # type: ignore
        [
            ("ndarray", np.array([1, 2, 3])),
            ("explicit_values", "float32", [2, 2], "dense", b"data"),
        ]
    )
    def test_valid_init_overloads_kwargs(self, name: str, *args: Any) -> None:
        """Ensure valid overloads initialize correctly."""
        if name == "explicit_values":
            array = Array(dtype=args[0], shape=args[1], stype=args[2], data=args[3])
        else:
            kwargs = {name: args[0]}
            array = Array(**kwargs)
        self.assertIsInstance(array, Array)

    @parameterized.expand(  # type: ignore
        [
            (np.array([1, 2, 3]),),
            ("float32", [2, 2], "dense", b"data"),
        ]
    )
    def test_valid_init_overloads_args(self, *args: Any) -> None:
        """Ensure valid overloads initialize correctly."""
        array = Array(*args)
        self.assertIsInstance(array, Array)

    @parameterized.expand(  # type: ignore
        [
            ("float32", [2, 2], "dense", 213),
            ([2, 2], "dense", b"data"),
            (123, "invalid"),
        ]
    )
    def test_invalid_init_combinations(self, *args: Any) -> None:
        """Ensure invalid combinations raise TypeError."""
        with self.assertRaises(TypeError):
            Array(*args)


@pytest.mark.parametrize(
    "shape, dtype",
    [
        ([100], "float32"),
        ([31, 31], "int8"),
        ([31, 153], "bool_"),  # bool_ is represented as a whole Byte in NumPy
    ],
)
def test_count_bytes(shape: list[int], dtype: str) -> None:
    """Test bytes in a ParametersRecord are computed correctly."""
    original_array = np.random.randn(*shape).astype(np.dtype(dtype))

    buff = ndarray_to_bytes(original_array)

    buffer = _get_buffer_from_ndarray(original_array)

    array_instance = Array(
        dtype=str(original_array.dtype),
        shape=list(original_array.shape),
        stype=SType.NUMPY,
        data=buffer,
    )
    key_name = "data"
    p_record = ParametersRecord(OrderedDict({key_name: array_instance}))

    assert len(buff) + len(key_name) == p_record.count_bytes()
