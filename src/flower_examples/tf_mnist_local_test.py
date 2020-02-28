# Copyright 2020 Adap GmbH. All Rights Reserved.
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
"""Tests for MNIST example."""


from .tf_mnist_local import load_model


def test_model_input_shape():
    """Test if the Keras model input shape is compatible with MNIST."""
    # Prepare
    expected = [None, 28, 28]
    model = load_model()

    # Execute
    actual = model.layers[0].get_input_at(0).get_shape().as_list()

    # Assert
    assert expected == actual
