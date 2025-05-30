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
"""Unit tests for Array."""


import sys
import unittest
from io import BytesIO
from types import ModuleType
from typing import Any
from unittest.mock import Mock

import numpy as np
from parameterized import parameterized

from flwr.common.serde import array_to_proto

from ..constant import SType
from ..inflatable import get_object_body, get_object_type_from_object_content
from ..typing import NDArray
from .array import Array


def _get_buffer_from_ndarray(array: NDArray) -> bytes:
    """Return a bytes buffer from a given NumPy array."""
    buffer = BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


class TorchTensor(Mock):
    """Mock Torch tensor class."""


MOCK_TORCH_TENSOR = TorchTensor(numpy=lambda: np.array([[1, 2, 3]]))
MOCK_TORCH_TENSOR.detach.return_value = MOCK_TORCH_TENSOR
MOCK_TORCH_TENSOR.cpu.return_value = MOCK_TORCH_TENSOR


class TestArray(unittest.TestCase):
    """Unit tests for Array."""

    def setUp(self) -> None:
        """Set up the test case."""
        # Patch torch
        self.torch_mock = Mock(spec=ModuleType, Tensor=TorchTensor)
        self._original_torch = sys.modules.get("torch")
        sys.modules["torch"] = self.torch_mock

    def tearDown(self) -> None:
        """Tear down the test case."""
        # Unpatch torch
        del sys.modules["torch"]
        if self._original_torch is not None:
            sys.modules["torch"] = self._original_torch

    def test_numpy_conversion_valid(self) -> None:
        """Test the numpy method with valid Array instance."""
        # Prepare
        original_array = np.array([1, 2, 3], dtype=np.float32)

        buffer = _get_buffer_from_ndarray(original_array)

        # Execute
        array_instance = Array(
            dtype=str(original_array.dtype),
            shape=tuple(original_array.shape),
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
            shape=(3,),
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
        self.assertEqual(array_instance.shape, tuple(original_array.shape))
        self.assertEqual(array_instance.stype, SType.NUMPY)
        np.testing.assert_array_equal(deserialized_array, original_array)

    def test_from_torch_tensor_with_torch(self) -> None:
        """Test creating an Array from a PyTorch tensor (mocked torch)."""
        # Prepare
        mock_tensor = TorchTensor()

        # Mock .detach().cpu().numpy() to return a NumPy array
        mock_tensor.detach.return_value = mock_tensor
        mock_tensor.cpu.return_value = mock_tensor
        mock_tensor.numpy.return_value = np.array([[5, 6], [7, 8]], dtype=np.float32)

        # Execute
        arr = Array.from_torch_tensor(mock_tensor)

        # Assert
        self.assertEqual(arr.dtype, "float32")
        self.assertEqual(arr.shape, (2, 2))
        self.assertEqual(arr.stype, SType.NUMPY)

    @parameterized.expand(  # type: ignore
        [
            ({"torch_tensor": MOCK_TORCH_TENSOR},),
            ({"ndarray": np.array([1, 2, 3])},),
            ({"dtype": "float32", "shape": (2, 2), "stype": "dense", "data": b"data"},),
        ]
    )
    def test_valid_init_overloads_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Ensure valid overloads initialize correctly."""
        array = Array(**kwargs)
        self.assertIsInstance(array, Array)

    @parameterized.expand(  # type: ignore
        [
            (MOCK_TORCH_TENSOR,),
            (np.array([1, 2, 3]),),
            ("float32", (2, 2), "dense", b"data"),
        ]
    )
    def test_valid_init_overloads_args(self, *args: Any) -> None:
        """Ensure valid overloads initialize correctly."""
        array = Array(*args)
        self.assertIsInstance(array, Array)

    @parameterized.expand(  # type: ignore
        [
            (MOCK_TORCH_TENSOR, np.array([1])),
            ("float32", (2, 2), "dense", 213),
            ((2, 2), "dense", b"data"),
            (123, "invalid"),
        ]
    )
    def test_invalid_init_combinations(self, *args: Any) -> None:
        """Ensure invalid combinations raise TypeError."""
        with self.assertRaises(TypeError):
            Array(*args)

    def test_deflate_and_inflate(self) -> None:
        """Ensure an Array can be (de)inflated correctly."""
        arr = Array(np.random.randn(5, 5))

        # Assert
        # Array has no children
        assert arr.children is None

        arr_b = arr.deflate()

        # Assert
        # Class name matches
        assert get_object_type_from_object_content(arr_b) == arr.__class__.__qualname__
        # Body of deflfated Array matches its direct protobuf serialization
        assert get_object_body(arr_b, Array) == array_to_proto(arr).SerializeToString()

        # Inflate
        arr_ = Array.inflate(arr_b)

        # Assert
        # Both objects are identical
        assert arr.object_id == arr_.object_id

        # Assert
        # Inflate passing children raises ValueError
        self.assertRaises(ValueError, Array.inflate, arr_b, children={"123": arr})
