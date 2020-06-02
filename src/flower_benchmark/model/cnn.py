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
"""CNN."""


from typing import Optional, Tuple

import tensorflow as tf

CNN_REG = 1e-5
DENSE_REG = 1e-3


def orig_cnn(
    input_shape: Tuple[int, int, int] = (28, 28, 1), seed: Optional[int] = None
) -> tf.keras.Model:
    """Create a CNN instance."""
    # Kernel initializer
    kernel_initializer = tf.keras.initializers.glorot_uniform(seed=seed)

    # Architecture
    inputs = tf.keras.layers.Input(shape=input_shape)
    layers = tf.keras.layers.Conv2D(
        32,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer=kernel_initializer,
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(CNN_REG),
    )(inputs)
    layers = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(layers)
    layers = tf.keras.layers.Conv2D(
        64,
        kernel_size=(5, 5),
        strides=(1, 1),
        kernel_initializer=kernel_initializer,
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(CNN_REG),
    )(layers)
    layers = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(layers)
    layers = tf.keras.layers.Flatten()(layers)
    layers = tf.keras.layers.Dense(
        512,
        kernel_initializer=kernel_initializer,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(DENSE_REG),
        bias_regularizer=tf.keras.regularizers.l2(DENSE_REG),
    )(layers)

    outputs = tf.keras.layers.Dense(
        10,
        kernel_initializer=kernel_initializer,
        activation="softmax",
        kernel_regularizer=tf.keras.regularizers.l2(DENSE_REG),
        bias_regularizer=tf.keras.regularizers.l2(DENSE_REG),
    )(layers)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile model w/ learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3, decay_steps=10000, decay_rate=0.9,
    )
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=["accuracy"],
    )
    return model


def keyword_cnn(
    input_shape: Tuple[int, int, int] = (80, 40, 1), seed: Optional[int] = None
) -> tf.keras.Model:
    """Create a keyword detection model instance."""
    # Kernel initializer
    kernel_initializer = tf.keras.initializers.glorot_uniform(seed=seed)

    # Architecture
    inputs = tf.keras.layers.Input(shape=input_shape)
    layers = tf.keras.layers.Conv2D(
        32,
        kernel_size=(20, 8),
        strides=(1, 1),
        kernel_initializer=kernel_initializer,
        padding="same",
        activation="relu",
    )(inputs)
    layers = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(layers)
    layers = tf.keras.layers.Dropout(0.5)(layers)
    layers = tf.keras.layers.Conv2D(
        64,
        kernel_size=(10, 4),
        strides=(1, 1),
        kernel_initializer=kernel_initializer,
        padding="same",
        activation="relu",
    )(layers)
    layers = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(layers)

    layers = tf.keras.layers.Conv2D(
        64,
        kernel_size=(2, 2),
        strides=(1, 1),
        kernel_initializer=kernel_initializer,
        padding="same",
        activation="relu",
    )(layers)

    layers = tf.keras.layers.GlobalAveragePooling2D()(layers)
    layers = tf.keras.layers.Dense(
        128, kernel_initializer=kernel_initializer, activation="relu"
    )(layers)

    outputs = tf.keras.layers.Dense(
        10, kernel_initializer=kernel_initializer, activation="softmax"
    )(layers)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile model w/ learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3, decay_steps=10000, decay_rate=0.9,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=["accuracy"],
    )

    return model
