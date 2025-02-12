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
from unittest.mock import Mock, call, patch

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


class TestParametersRecord(unittest.TestCase):
    """Unit tests for ParametersRecord."""

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

    @parameterized.expand(  # type: ignore
        [
            ([np.array([1, 2]), np.array([3, 4])],),  # Two arrays
            ([np.array(5)],),  # Single array
            ([],),  # Empty list
        ]
    )
    def test_from_numpy_ndarrays(self, ndarrays: list[NDArray]) -> None:
        """Test creating a ParametersRecord from a list of NumPy arrays."""
        with patch.object(Array, "from_numpy_ndarray") as mock_from_numpy:
            # Prepare
            mock_arrays = [Mock(spec=Array) for _ in ndarrays]
            mock_from_numpy.side_effect = mock_arrays
            expected_keys = [str(i) for i in range(len(ndarrays))]

            # Execute
            record = ParametersRecord.from_numpy_ndarrays(ndarrays)

            # Assert
            self.assertEqual(list(record.keys()), expected_keys)
            self.assertEqual(list(record.values()), mock_arrays)
            mock_from_numpy.assert_has_calls(
                [call(arr) for arr in ndarrays], any_order=False
            )

    def test_from_state_dict_with_torch(self) -> None:
        """Test creating a ParametersRecord from a PyTorch state_dict."""
        # Prepare
        # Mock state_dict with tensor mocks
        state_dict = OrderedDict(
            [
                ("weight", TorchTensor()),
                ("bias", TorchTensor()),
            ]
        )
        ndarrays = [np.array([1, 2]), np.array([3, 4])]
        for tensor_mock, numpy_array in zip(state_dict.values(), ndarrays):
            tensor_mock.detach.return_value = tensor_mock
            tensor_mock.cpu.return_value = tensor_mock
            tensor_mock.numpy.return_value = numpy_array

        # Mock Array.from_numpy_ndarray to return mock arrays
        mock_arrays = [Mock(spec=Array), Mock(spec=Array)]
        with patch.object(Array, "from_numpy_ndarray") as mock_from_numpy:
            mock_from_numpy.side_effect = mock_arrays

            # Execute
            record = ParametersRecord.from_state_dict(state_dict)

            # Assert
            self.assertEqual(list(record.keys()), list(state_dict.keys()))
            for tensor_mock in state_dict.values():
                tensor_mock.detach.assert_called_once()
                tensor_mock.cpu.assert_called_once()
                tensor_mock.numpy.assert_called_once()
            mock_from_numpy.assert_has_calls(
                [call(arr) for arr in ndarrays], any_order=False
            )
            self.assertEqual(list(record.values()), mock_arrays)

    def test_from_state_dict_without_torch(self) -> None:
        """Test `ParametersRecord.from_state_dict` without PyTorch."""
        with patch.dict("sys.modules", {}, clear=True):
            with self.assertRaises(RuntimeError) as cm:
                ParametersRecord.from_state_dict(OrderedDict())
            self.assertIn("PyTorch is required", str(cm.exception))

    @parameterized.expand(  # type: ignore
        [
            ([np.array([1, 2]), np.array([3, 4])],),  # Two arrays
            ([np.array(5)],),  # Single array
            ([],),  # Empty list
        ]
    )
    def test_from_tf_weights(self, tf_weights: list[NDArray]) -> None:
        """Test creating a ParametersRecord from a list of TensorFlow weights."""
        with patch.object(Array, "from_numpy_ndarray") as mock_from_numpy:
            mock_from_numpy.return_value = Mock(spec=Array)
            ParametersRecord.from_tf_weights(tf_weights)
            mock_from_numpy.assert_has_calls(
                [call(w) for w in tf_weights], any_order=False
            )

    def test_to_numpy_ndarrays(self) -> None:
        """Test converting a ParametersRecord to a list of NumPy arrays."""
        # Prepare
        record = ParametersRecord()
        numpy_arrays = [np.array([1, 2]), np.array([3, 4])]
        mock_arrays = [Mock(spec=Array), Mock(spec=Array)]
        for mock_arr, arr in zip(mock_arrays, numpy_arrays):
            mock_arr.numpy.return_value = arr
        record["0"] = mock_arrays[0]
        record["1"] = mock_arrays[1]

        # Execute
        result = record.to_numpy_ndarrays()

        # Assert
        self.assertEqual(result, numpy_arrays)
        for mock_arr in mock_arrays:
            mock_arr.numpy.assert_called_once()

    def test_to_state_dict_with_torch(self) -> None:
        """Test converting a ParametersRecord to a PyTorch state_dict."""
        # Prepare
        # Mock torch.from_numpy method
        tensors = [TorchTensor(), TorchTensor()]
        self.torch_mock.from_numpy = Mock(side_effect=tensors)
        record = ParametersRecord()
        ndarrays = [np.array([1, 2]), np.array([3, 4])]
        mock_arrays = [Mock(spec=Array), Mock(spec=Array)]
        for mock_arr, arr in zip(mock_arrays, ndarrays):
            mock_arr.numpy.return_value = arr
        record["weight"] = mock_arrays[0]
        record["bias"] = mock_arrays[1]

        # Execute
        state_dict = record.to_state_dict()

        # Assert
        self.assertIsInstance(state_dict, OrderedDict)
        self.assertEqual(list(state_dict.keys()), ["weight", "bias"])
        self.torch_mock.from_numpy.assert_has_calls(
            [call(arr) for arr in ndarrays], any_order=False
        )
        self.assertEqual(list(state_dict.values()), tensors)

    def test_to_state_dict_without_torch(self) -> None:
        """Test `ParametersRecord.to_state_dict` without PyTorch."""
        with patch.dict("sys.modules", {}, clear=True):
            record = ParametersRecord()
            with self.assertRaises(RuntimeError) as cm:
                record.to_state_dict()
            self.assertIn("PyTorch is required", str(cm.exception))

    def test_to_tf_weights(self) -> None:
        """Test converting a ParametersRecord to a list of TensorFlow weights."""
        # Prepare
        record = ParametersRecord()
        numpy_arrays = [np.array([1, 2]), np.array([3, 4])]
        mock_arrays = [Mock(spec=Array), Mock(spec=Array)]
        for mock_arr, arr in zip(mock_arrays, numpy_arrays):
            mock_arr.numpy.return_value = arr
        record["0"] = mock_arrays[0]
        record["1"] = mock_arrays[1]

        # Execute
        result = record.to_tf_weights()

        # Assert
        self.assertEqual(result, numpy_arrays)
        for mock_arr in mock_arrays:
            mock_arr.numpy.assert_called_once()

    def test_init_no_args(self) -> None:
        """Test initializing with no arguments."""
        _ = ParametersRecord()

    @parameterized.expand(  # type: ignore
        [
            ([np.array([1, 2, 3])], True),
            ([np.array([1, 2, 3])], False),
            ([np.array([4, 5, 6]), np.array([1, 2, 3])], True),
            ([np.array([4, 5, 6]), np.array([1, 2, 3])], False),
        ]
    )
    def test_init_ndarrays_calls_method(
        self, ndarrays: list[NDArray], use_keyword: bool
    ) -> None:
        """Test initializing with NumPy arrays."""
        with patch.object(
            ParametersRecord,
            "from_numpy_ndarrays",
            return_value=Mock(spec=ParametersRecord),
        ) as mock_from_numpy:
            if use_keyword:
                _ = ParametersRecord(numpy_ndarrays=ndarrays)
            else:
                _ = ParametersRecord(ndarrays)
            mock_from_numpy.assert_called_once_with(ndarrays)

    @parameterized.expand([(True,), (False,)])  # type: ignore
    def test_init_array_dict_keep_input_false(self, use_keyword: bool) -> None:
        """Test initializing with an array_dict and keep_input=False."""
        # Prepare
        mock_arr = Mock(spec=Array)
        arr_dict: OrderedDict[str, Array] = OrderedDict({"x": mock_arr})

        # Execute
        if use_keyword:
            record = ParametersRecord(array_dict=arr_dict, keep_input=False)
        else:
            record = ParametersRecord(arr_dict, keep_input=False)

        # Assert
        self.assertEqual(record["x"], mock_arr)
        self.assertEqual(len(arr_dict), 0)

    @parameterized.expand([(True,), (False,)])  # type: ignore
    def test_init_array_dict_keep_input_true(self, use_keyword: bool) -> None:
        """Test initializing with an array_dict and keep_input=True."""
        # Prepare
        arr_dict: OrderedDict[str, Array] = OrderedDict({"x": Mock(spec=Array)})

        # Execute
        if use_keyword:
            record = ParametersRecord(array_dict=arr_dict, keep_input=True)
        else:
            record = ParametersRecord(arr_dict, keep_input=True)

        # Assert
        self.assertEqual(record["x"], arr_dict["x"])
        self.assertEqual(len(arr_dict), 1)

    @parameterized.expand([(True,), (False,)])  # type: ignore
    def test_init_state_dict_calls_from_state_dict(self, use_keyword: bool) -> None:
        """Test initializing with a state_dict."""
        state_dict = OrderedDict({"layer.weight": MOCK_TORCH_TENSOR})
        with patch.object(
            ParametersRecord,
            "from_state_dict",
            return_value=Mock(spec=ParametersRecord),
        ) as mock_from_state_dict:
            if use_keyword:
                _ = ParametersRecord(state_dict=state_dict)
            else:
                _ = ParametersRecord(state_dict)

            # from_state_dict() should be called exactly once with the provided dict
            mock_from_state_dict.assert_called_once_with(state_dict)

    @parameterized.expand(  # type: ignore
        [
            ((42,), {}),
            (("invalid",), {}),
            (
                (),
                {"numpy_ndarrays": [np.array([2])], "tf_weights": [np.array([2])]},
            ),
            (([np.array([1])],), {"array_dict": {"x": Mock(spec=Array)}}),
            (([np.array([1])],), {"numpy_ndarrays": [np.array([2])]}),
            (([np.array([1])],), {"state_dict": {"layer.weight": MOCK_TORCH_TENSOR}}),
            (([np.array([1])],), {"tf_weights": [np.array([2])]}),
        ]
    )
    def test_init_unrecognized_arg_raises_error(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        """Test initializing with unrecognized arguments."""
        with self.assertRaisesRegex(
            TypeError, "Invalid arguments for ParametersRecord.*"
        ):
            ParametersRecord(*args, **kwargs)


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
