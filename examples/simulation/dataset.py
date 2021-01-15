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
"""Partitioned version of CIFAR-10 dataset."""

from typing import List, Tuple, cast

import numpy as np
import tensorflow as tf

XY = Tuple[np.ndarray, np.ndarray]
XYList = List[XY]
PartitionedDataset = List[Tuple[XY, XY]]


def shuffle(x: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle x and y."""
    idx = np.random.permutation(len(x))
    return x[idx], y[idx]


def partition(x: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split x and y into a number of partitions."""
    return list(zip(np.split(x, num_partitions), np.split(y, num_partitions)))


def create_partitions(
    source_dataset: XY,
    num_partitions: int,
) -> XYList:
    """Create partitioned version of a source dataset."""
    x, y = source_dataset
    x, y = shuffle(x, y)
    xy_partitions = partition(x, y, num_partitions)

    return xy_partitions


def load(
    num_partitions: int,
) -> PartitionedDataset:
    """Create partitioned version of CIFAR-10."""
    xy_train, xy_test = tf.keras.datasets.cifar10.load_data()

    xy_train_partitions = create_partitions(xy_train, num_partitions)
    xy_test_partitions = create_partitions(xy_test, num_partitions)

    return list(zip(xy_train_partitions, xy_test_partitions))
