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
"""Flower client using TensorFlow/Keras for image classification."""


from logging import DEBUG
from typing import Tuple

import numpy as np
import tensorflow as tf

import flwr as fl
from flwr.common.logger import log

from .common import custom_fit, keras_evaluate
from .data import build_dataset

tf.get_logger().setLevel("ERROR")


class VisionClassificationClient(fl.client.Client):
    """Flower client implementing image classification using TensorFlow/Keras."""

    # pylint: disable-msg=too-many-arguments
    def __init__(
        self,
        cid: str,
        model: tf.keras.Model,
        xy_train: Tuple[np.ndarray, np.ndarray],
        xy_test: Tuple[np.ndarray, np.ndarray],
        delay_factor: float,
        num_classes: int,
        augment: bool = False,
        augment_horizontal_flip: bool = False,
        augment_offset: int = 0,
        normalization_factor: float = 255.0,
    ):
        super().__init__(cid)
        self.model = model
        self.ds_train = build_dataset(
            xy_train[0],
            xy_train[1],
            num_classes=num_classes,
            shuffle_buffer_size=len(xy_train[0]),
            augment=augment,
            augment_horizontal_flip=augment_horizontal_flip,
            augment_offset=augment_offset,
            normalization_factor=normalization_factor,
        )
        self.ds_test = build_dataset(
            xy_test[0],
            xy_test[1],
            num_classes=num_classes,
            shuffle_buffer_size=0,
            augment=False,
            normalization_factor=normalization_factor,
        )
        self.num_examples_train = len(xy_train[0])
        self.num_examples_test = len(xy_test[0])
        self.delay_factor = delay_factor

    def get_parameters(self) -> fl.common.ParametersRes:
        parameters = fl.common.weights_to_parameters(self.model.get_weights())
        return fl.common.ParametersRes(parameters=parameters)

    def fit(self, ins: fl.common.FitIns) -> fl.common.FitRes:
        weights: fl.common.Weights = fl.common.parameters_to_weights(ins[0])
        config = ins[1]
        log(
            DEBUG,
            "fit on %s (examples: %s), config %s",
            self.cid,
            self.num_examples_train,
            config,
        )

        # Training configuration
        # epoch_global = int(config["epoch_global"])
        epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])
        # lr_initial = float(config["lr_initial"])
        # lr_decay = float(config["lr_decay"])
        timeout = int(config["timeout"]) if "timeout" in config else None
        partial_updates = bool(int(config["partial_updates"]))

        # Use provided weights to update the local model
        self.model.set_weights(weights)

        # Train the local model using the local dataset
        completed, fit_duration, num_examples = custom_fit(
            model=self.model,
            dataset=self.ds_train,
            num_epochs=epochs,
            batch_size=batch_size,
            callbacks=[],
            delay_factor=self.delay_factor,
            timeout=timeout,
        )
        log(DEBUG, "client %s had fit_duration %s", self.cid, fit_duration)

        # Compute the maximum number of examples which could have been processed
        num_examples_ceil = self.num_examples_train * epochs

        if not completed and not partial_updates:
            # Return empty update if local update could not be completed in time
            parameters = fl.common.weights_to_parameters([])
        else:
            # Return the refined weights and the number of examples used for training
            parameters = fl.common.weights_to_parameters(self.model.get_weights())
        return parameters, num_examples, num_examples_ceil, fit_duration

    def evaluate(self, ins: fl.common.EvaluateIns) -> fl.common.EvaluateRes:
        weights = fl.common.parameters_to_weights(ins[0])
        config = ins[1]
        log(
            DEBUG,
            "evaluate on %s (examples: %s), config %s",
            self.cid,
            self.num_examples_test,
            config,
        )

        # Use provided weights to update the local model
        self.model.set_weights(weights)

        # Evaluate the updated model on the local dataset
        loss, acc = keras_evaluate(
            self.model, self.ds_test, batch_size=self.num_examples_test
        )

        # Return the number of evaluation examples and the evaluation result (loss)
        return self.num_examples_test, loss, acc
