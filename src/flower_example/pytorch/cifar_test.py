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
"""Tests for PyTorch CIFAR-10 image classification."""

import unittest

import numpy as np

import flower as fl

from . import cifar


class CifarTestCase(unittest.TestCase):
    """Tests for cifar module."""

    def test_load_model(self):
        """Test the number of (trainable) model parameters."""
        # pylint: disable-msg=no-self-use

        # Prepare
        expected = 62006

        # Execute
        model: cifar.Net = cifar.load_model()
        actual = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Assert
        assert actual == expected

    def test_get_weights(self):
        """Test get_weights."""
        # pylint: disable-msg=no-self-use

        # Prepare
        model: cifar.Net = cifar.load_model()
        expected = 10

        # Execute
        weights: fl.Weights = model.get_weights()

        # Assert
        assert len(weights) == expected

    def test_set_weights(self):
        """Test set_weights."""
        # pylint: disable-msg=no-self-use

        # Prepare
        weights_expected: fl.Weights = cifar.load_model().get_weights()
        model: cifar.Net = cifar.load_model()

        # Execute
        model.set_weights(weights_expected)
        weights_actual: fl.Weights = model.get_weights()

        # Assert
        for nda_expected, nda_actual in zip(weights_expected, weights_actual):
            np.testing.assert_array_equal(nda_expected, nda_actual)


if __name__ == "__main__":
    unittest.main(verbosity=2)
