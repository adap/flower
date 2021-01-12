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
"""Common baseline components."""


import time
import timeit
from logging import INFO
from typing import Callable, List, Optional, Tuple

import numpy as np
import tensorflow as tf

import flwr as fl
from flwr.common.logger import log

from .data import build_dataset


# pylint: disable=unused-argument,invalid-name,too-many-arguments,too-many-locals
def custom_fit(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    num_epochs: int,
    batch_size: int,
    callbacks: List[tf.keras.callbacks.Callback],
    delay_factor: float = 0.0,
    timeout: Optional[int] = None,
) -> Tuple[bool, float, int]:
    """Train the model using a custom training loop."""
    ds_train = dataset.batch(batch_size=batch_size, drop_remainder=False)

    # Keep results for plotting
    train_loss_results = []
    train_accuracy_results = []

    # Optimizer
    optimizer = tf.keras.optimizers.Adam()

    fit_begin = timeit.default_timer()
    num_examples = 0
    for epoch in range(num_epochs):
        log(INFO, "Starting epoch %s", epoch)

        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

        # Single loop over the dataset
        batch_begin = timeit.default_timer()
        num_examples_batch = 0
        for batch, (x, y) in enumerate(ds_train):
            num_examples_batch += len(x)

            # Optimize the model
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add the current batch loss
            epoch_accuracy.update_state(y, model(x, training=True))

            # Track the number of examples used for training
            num_examples += x.shape[0]

            # Delay
            batch_duration = timeit.default_timer() - batch_begin
            if delay_factor > 0.0:
                time.sleep(batch_duration * delay_factor)

            # Progress log
            if batch % 100 == 0:
                log(
                    INFO,
                    "Batch %s: loss %s (%s examples processed, batch duration: %s)",
                    batch,
                    loss_value,
                    num_examples_batch,
                    batch_duration,
                )

            # Timeout
            if timeout is not None:
                fit_duration = timeit.default_timer() - fit_begin
                if fit_duration > timeout:
                    log(INFO, "client timeout")
                    return (False, fit_duration, num_examples)
            batch_begin = timeit.default_timer()

    # End epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())
    log(
        INFO,
        "Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(
            epoch, epoch_loss_avg.result(), epoch_accuracy.result()
        ),
    )

    fit_duration = timeit.default_timer() - fit_begin
    return True, fit_duration, num_examples


def loss(
    model: tf.keras.Model, x: tf.Tensor, y: tf.Tensor, training: bool
) -> tf.Tensor:
    """Calculate categorical crossentropy loss."""
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    y_ = model(x, training=training)
    return loss_object(y_true=y, y_pred=y_)


def grad(
    model: tf.keras.Model, x: tf.Tensor, y: tf.Tensor
) -> Tuple[tf.Tensor, List[tf.Tensor]]:
    """Calculate gradients."""
    with tf.GradientTape() as tape:
        loss_value = loss(model, x, y, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def keras_evaluate(
    model: tf.keras.Model, dataset: tf.data.Dataset, batch_size: int
) -> Tuple[float, float]:
    """Evaluate the model using model.evaluate(...)."""
    ds_test = dataset.batch(batch_size=batch_size, drop_remainder=False)
    test_loss, acc = model.evaluate(x=ds_test)
    return float(test_loss), float(acc)


def keras_fit(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    num_epochs: int,
    batch_size: int,
    callbacks: List[tf.keras.callbacks.Callback],
) -> None:
    """Train the model using model.fit(...)."""
    ds_train = dataset.batch(batch_size=batch_size, drop_remainder=False)
    model.fit(ds_train, epochs=num_epochs, callbacks=callbacks, verbose=2)


def get_lr_schedule(
    epoch_global: int, lr_initial: float, lr_decay: float
) -> Callable[[int], float]:
    """Return a schedule which decays the learning rate after each epoch."""

    def lr_schedule(epoch: int) -> float:
        """Learning rate schedule."""
        epoch += epoch_global
        return lr_initial * lr_decay ** epoch

    return lr_schedule


def get_eval_fn(
    model: tf.keras.Model, num_classes: int, xy_test: Tuple[np.ndarray, np.ndarray]
) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    ds_test = build_dataset(
        xy_test[0],
        xy_test[1],
        num_classes=num_classes,
        shuffle_buffer_size=0,
        augment=False,
    )

    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        """Use entire test set for evaluation."""
        model.set_weights(weights)
        lss, acc = keras_evaluate(model, ds_test, batch_size=len(xy_test[0]))
        return lss, acc

    return evaluate
