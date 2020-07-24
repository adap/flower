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
"""Baseline utilities for data loading."""


from typing import List, Optional, Tuple, cast

import numpy as np
import tensorflow as tf


# pylint: disable-msg=too-many-arguments
def load_partition(
    xy_partitions: List[Tuple[np.ndarray, np.ndarray]],
    xy_test: Tuple[np.ndarray, np.ndarray],
    partition: int,
    num_clients: int,
    seed: int,
    dry_run: bool = False,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load, normalize, and sample CIFAR-10/100."""

    # Take partition
    x_train, y_train = xy_partitions[partition]

    # Take a subset of the test set
    x_test, y_test = shuffle(xy_test[0], xy_test[1], seed=seed)
    x_test, y_test = get_partition(x_test, y_test, partition, num_clients)

    # Adjust x shape for model
    if x_train.ndim == 3:
        x_train = adjust_x_shape(x_train)
        x_test = adjust_x_shape(x_test)

    # Adjust y shape for model
    if y_train.ndim == 2:
        y_train = adjust_y_shape(y_train)
        y_test = adjust_y_shape(y_test)

    # Return a small subset of the data if dry_run is set
    if dry_run:
        return (x_train[0:100], y_train[0:100]), (x_test[0:50], y_test[0:50])
    return (x_train, y_train), (x_test, y_test)


def shuffle(
    x_orig: np.ndarray, y_orig: np.ndarray, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Shuffle x and y in the same way."""
    np.random.seed(seed)
    idx = np.random.permutation(len(x_orig))
    return x_orig[idx], y_orig[idx]


def get_partition(
    x_orig: np.ndarray, y_orig: np.ndarray, partition: int, num_clients: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a single partition of an equally partitioned dataset."""
    step_size = len(x_orig) / num_clients
    start_index = int(step_size * partition)
    end_index = int(start_index + step_size)
    return x_orig[start_index:end_index], y_orig[start_index:end_index]


def adjust_x_shape(nda: np.ndarray) -> np.ndarray:
    """Turn shape (x, y, z) into (x, y, z, 1)."""
    nda_adjusted = np.reshape(nda, (nda.shape[0], nda.shape[1], nda.shape[2], 1))
    return cast(np.ndarray, nda_adjusted)


def adjust_y_shape(nda: np.ndarray) -> np.ndarray:
    """Turn shape (x, 1) into (x)."""
    nda_adjusted = np.reshape(nda, (nda.shape[0]))
    return cast(np.ndarray, nda_adjusted)


# pylint: disable-msg=too-many-arguments,invalid-name
def build_dataset(
    x: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    shuffle_buffer_size: int = 0,
    augment: bool = False,
    augment_color: bool = False,
    augment_horizontal_flip: bool = False,
    augment_offset: int = 0,
    seed: Optional[int] = None,
    normalization_factor: float = 255.0,
) -> tf.data.Dataset:
    """Normalize images, one-hot encode labels, optionally shuffle and augment."""
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(
        lambda x, y: (
            tf.cast(x, tf.float32) / normalization_factor,
            tf.one_hot(
                indices=tf.cast(y, tf.int32), depth=num_classes, on_value=1, off_value=0
            ),
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(
            buffer_size=shuffle_buffer_size, seed=seed, reshuffle_each_iteration=True
        )
    if augment:
        dataset = dataset.map(
            lambda x, y: (
                apply_augmentation(
                    x,
                    seed=seed,
                    color=augment_color,
                    horizontal_flip=augment_horizontal_flip,
                    offset=augment_offset,
                ),
                y,
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    return dataset


def apply_augmentation(
    img: tf.Tensor,
    seed: Optional[int],
    color: bool,
    horizontal_flip: bool,
    offset: int,
) -> tf.Tensor:
    """Apply different augmentations to a single example."""
    if color:
        img = tf.image.random_hue(img, 0.08, seed=seed)
        img = tf.image.random_saturation(img, 0.6, 1.6, seed=seed)
        img = tf.image.random_brightness(img, 0.05, seed=seed)
        img = tf.image.random_contrast(img, 0.7, 1.3, seed=seed)
    if horizontal_flip:
        img = tf.image.random_flip_left_right(img, seed=seed)
    # Get image size from tensor
    size = img.shape.as_list()  # E.g., [28, 28, 1] or [32, 32, 3]
    height = size[0]
    width = size[1]
    img_padded = tf.image.pad_to_bounding_box(
        img, offset, offset, height + 2 * offset, width + 2 * offset
    )
    return tf.image.random_crop(img_padded, size=size, seed=seed)
