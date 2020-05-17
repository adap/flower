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
"""Benchmark utilities for data loading."""


from typing import Optional

import numpy as np
import tensorflow as tf


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
) -> tf.data.Dataset:
    """Divide images by 255, one-hot encode labels, optionally shuffle and augment."""
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(
        lambda x, y: (
            tf.cast(x, tf.float32) / 255.0,
            tf.one_hot(indices=tf.cast(y, tf.int32), depth=num_classes),
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
