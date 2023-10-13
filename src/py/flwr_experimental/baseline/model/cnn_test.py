# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
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
"""Tests for CNN models."""


from .cnn import orig_cnn


def test_cnn_size_mnist() -> None:
    """Test number of parameters with MNIST-sized inputs."""
    # Prepare
    model = orig_cnn(input_shape=(28, 28, 1))
    expected = 1_663_370

    # Execute
    actual = model.count_params()

    # Assert
    assert actual == expected


def test_cnn_size_cifar() -> None:
    """Test number of parameters with CIFAR-sized inputs."""
    # Prepare
    model = orig_cnn(input_shape=(32, 32, 3))
    expected = 2_156_490

    # Execute
    actual = model.count_params()

    # Assert
    assert actual == expected
