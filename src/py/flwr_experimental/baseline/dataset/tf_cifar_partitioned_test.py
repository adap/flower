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
"""Tests for partitioned CIFAR-10/100 dataset generation."""
# pylint: disable=no-self-use

import unittest

from flwr_experimental.baseline.dataset.tf_cifar_partitioned import load_data


class CifarPartitionedTestCase(unittest.TestCase):
    """Tests for partitioned CIFAR-10/100 dataset generation."""

    def test_load_data_integration(self) -> None:
        """Test partition function."""
        # Execute
        for num_partitions in [10, 100]:
            for fraction in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                (_, _), _ = load_data(fraction, num_partitions)


if __name__ == "__main__":
    unittest.main(verbosity=2)
