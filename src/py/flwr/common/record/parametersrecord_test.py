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


import sys
import unittest
from collections import OrderedDict
from io import BytesIO
from types import ModuleType
from typing import Any
from unittest.mock import Mock

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


class TorchTensor(Mock):
    """Mock Torch tensor class."""


class TfTensor(Mock):
    """Mock TensorFlow tensor class."""


MOCK_TORCH_TENSOR = TorchTensor(numpy=lambda: np.array([[1, 2, 3]]))
MOCK_TF_TENSOR = TfTensor(numpy=lambda: np.array([1, 2, 3]))
MOCK_TORCH_TENSOR.detach.return_value = MOCK_TORCH_TENSOR
MOCK_TORCH_TENSOR.cpu.return_value = MOCK_TORCH_TENSOR


class TestArray(unittest.TestCase):
    """Unit tests for Array."""

    def setUp(self) -> None:
        """Set up the test case."""
        # Patch torch and tensorflow
        self.torch_mock = Mock(spec=ModuleType, Tensor=TorchTensor)
        self.tf_mock = Mock(spec=ModuleType, Tensor=TfTensor)
        self._original_torch = sys.modules.get("torch")
        self._original_tf = sys.modules.get("tensorflow")
        sys.modules["torch"] = self.torch_mock
        sys.modules["tensorflow"] = self.tf_mock

    def tearDown(self) -> None:
        """Tear down the test case."""
        # Unpatch torch and tensorflow
        del sys.modules["torch"]
        del sys.modules["tensorflow"]
        if self._original_torch is not None:
            sys.modules["torch"] = self._original_torch
        if self._original_tf is not None:
            sys.modules["tensorflow"] = self._original_tf

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
        self.assertEqual(arr.shape, [2, 2])
        self.assertEqual(arr.stype, SType.NUMPY)

    def test_from_tf_tensor_with_tf(self) -> None:
        """Test creating an Array from a TensorFlow tensor (mocked tf)."""
        # Prepare
        mock_tensor = TfTensor()

        # Mock .numpy() to return a NumPy array
        mock_tensor.numpy.return_value = np.array([[9, 10], [11, 12]], dtype=np.float32)

        # Execute
        arr = Array.from_tf_tensor(mock_tensor)

        # Assert
        self.assertEqual(arr.dtype, "float32")
        self.assertEqual(arr.shape, [2, 2])
        self.assertEqual(arr.stype, SType.NUMPY)

    @parameterized.expand(  # type: ignore
        [
            ({"torch_tensor": MOCK_TORCH_TENSOR},),
            ({"tf_tensor": MOCK_TF_TENSOR},),
            ({"ndarray": np.array([1, 2, 3])},),
            ({"dtype": "float32", "shape": [2, 2], "stype": "dense", "data": b"data"},),
        ]
    )
    def test_valid_init_overloads_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Ensure valid overloads initialize correctly."""
        array = Array(**kwargs)
        self.assertIsInstance(array, Array)

    @parameterized.expand(  # type: ignore
        [
            (MOCK_TORCH_TENSOR,),
            (MOCK_TF_TENSOR,),
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
            (MOCK_TORCH_TENSOR, MOCK_TF_TENSOR),
            (MOCK_TORCH_TENSOR, np.array([1])),
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
