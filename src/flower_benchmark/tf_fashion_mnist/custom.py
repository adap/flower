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
"""Custom client-side training."""


import time
import timeit
from typing import List, Optional, Tuple

import tensorflow as tf


# pylint: disable-msg=unused-argument,invalid-name
def custom_fit(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    num_epochs: int,
    batch_size: int,
    callbacks: List[tf.keras.callbacks.callbacks.Callback],
    delay_factor: float = 0.0,
    timeout: Optional[int] = None,
) -> Tuple[bool, float]:
    """Train the model using a custom training loop."""
    ds_train = dataset.batch(batch_size=batch_size, drop_remainder=False)

    # Keep results for plotting
    train_loss_results = []
    train_accuracy_results = []

    # Optimizer
    optimizer = tf.keras.optimizers.Adam()

    fit_begin = timeit.default_timer()
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch}")

        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

        # Single loop over the dataset
        batches = 0
        batch_begin = timeit.default_timer()
        for x, y in ds_train:
            batches += 1

            # Optimize the model
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add the current batch loss
            epoch_accuracy.update_state(y, model(x, training=True))

            # Delay
            batch_duration = timeit.default_timer() - batch_begin
            if delay_factor > 0.0:
                time.sleep(batch_duration * delay_factor)
            if timeout is not None:
                fit_duration = timeit.default_timer() - fit_begin
                if fit_duration > timeout:
                    print("TIMEOUT!!!!1111!!!")
                    return (
                        False,
                        fit_duration,
                    )  # FIXME: also return num examples (see below)
            batch_begin = timeit.default_timer()

        print(f"Trained for {batches} batches during epoch {epoch}")

    # End epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())
    print(
        "Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(
            epoch, epoch_loss_avg.result(), epoch_accuracy.result()
        )
    )  # FIXME use log

    # FIXME collect and return the actual number of examples (incl. remainders)
    fit_duration = timeit.default_timer() - fit_begin
    return True, fit_duration


def loss(model: tf.keras.Model, x, y, training: bool) -> float:
    """Calculate categorical crossentropy loss."""
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    y_ = model(x, training=training)
    return loss_object(y_true=y, y_pred=y_)


def grad(model: tf.keras.Model, inputs, targets):
    """Calculate gradients."""
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)
