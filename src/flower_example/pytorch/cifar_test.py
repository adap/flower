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

from . import cifar


class CifarTestCase(unittest.TestCase):
    """Tests for cifar module."""

    def test_load_model(self):
        """Test the number of (trainable) model parameters."""
        # pylint: disable-msg=no-self-use

        # Prepare
        expected = 62006

        # Execute
        model = cifar.load_model()
        actual = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Assert
        assert actual == expected


if __name__ == "__main__":
    unittest.main(verbosity=2)
