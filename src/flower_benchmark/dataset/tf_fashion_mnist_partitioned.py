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
"""Partitioned versions of CIFAR-10/100 datasets."""
# pylint: disable=invalid-name


from typing import Tuple

import tensorflow as tf

from .dataset import (
    XY,
    PartitionedDataset,
    create_partitioned_dataset,
    log_distribution,
)


def load_data(
    iid_fraction: float, num_partitions: int
) -> Tuple[PartitionedDataset, XY]:
    """Load partitioned version of FashionMNIST."""
    (xy_train_partitions, xy_test_partitions), xy_test = create_partitioned_dataset(
        tf.keras.datasets.fashion_mnist.load_data(), iid_fraction, num_partitions
    )
    return (xy_train_partitions, xy_test_partitions), xy_test


if __name__ == "__main__":
    # Load a partitioned dataset and show distribution of examples
    for _num_partitions in [10, 100]:
        for _fraction in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            (xy_train_par, xy_test_par), _ = load_data(_fraction, _num_partitions)
            print(f"\nfraction: {_fraction}; num_partitions: {_num_partitions}")
            log_distribution(xy_train_par)
            log_distribution(xy_test_par)
