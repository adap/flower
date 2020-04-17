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
"""Create and augment a CIFAR-10 TensorFlow Dataset."""


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
    seed: Optional[int] = None,
) -> tf.data.Dataset:
    """Divide images by 255, one-hot encode labels, optionally shuffle and augment."""
    tf_ds = tf.data.Dataset.from_tensor_slices((x, y))
    tf_ds = tf_ds.map(
        lambda x, y: (
            tf.cast(x, tf.float32) / 255.0,
            tf.one_hot(indices=tf.cast(y, tf.int64), depth=num_classes, axis=1),
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    if shuffle_buffer_size > 0:
        tf_ds = tf_ds.shuffle(
            buffer_size=shuffle_buffer_size, seed=seed, reshuffle_each_iteration=True
        )
    if augment:
        tf_ds = tf_ds.map(
            lambda x, y: (_augment(x, seed=seed), y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    return tf_ds


def _augment(
    img: tf.Tensor,
    seed: Optional[int],
    color: bool = True,
    horizontal_flip: bool = True,
) -> tf.Tensor:
    if color:
        img = tf.image.random_hue(img, 0.08, seed=seed)
        img = tf.image.random_saturation(img, 0.6, 1.6, seed=seed)
        img = tf.image.random_brightness(img, 0.05, seed=seed)
        img = tf.image.random_contrast(img, 0.7, 1.3, seed=seed)
    if horizontal_flip:
        img = tf.image.random_flip_left_right(img, seed=seed)
    img_padded = tf.image.pad_to_bounding_box(img, 4, 4, 40, 40)
    return tf.image.random_crop(img_padded, size=[32, 32, 3], seed=seed)
