"""tf_privacy: Training with Sample-Level Differential Privacy using TensorFlow-Privacy Engine."""

import argparse
import os

import tensorflow as tf
import tensorflow_privacy
from flwr.client import ClientApp, NumPyClient
from flwr_datasets import FederatedDataset
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import (
    compute_dp_sgd_privacy_statement,
)
from flwr.common import Context

from tf_privacy.task import load_data, Net
import numpy as np


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class FlowerClient(NumPyClient):
    def __init__(
        self,
        model,
        train_data,
        test_data,
        noise_multiplier,
        run_config,
    ) -> None:
        super().__init__()
        self.model = model
        self.x_train, self.y_train = train_data
        self.x_train = np.expand_dims(self.x_train, axis=-1)
        self.x_test, self.y_test = test_data
        self.x_test = np.expand_dims(self.x_test, axis=-1)
        self.noise_multiplier = noise_multiplier
        if run_config["batch-size"] % run_config["num-microbatches"] != 0:
            raise ValueError(
                f"Batch size {run_config['batch-size']} is not divisible by the number of microbatches {run_config['num-microbatches']}"
            )

        self.optimizer = tensorflow_privacy.DPKerasSGDOptimizer(
            l2_norm_clip=run_config["l2-norm-clip"],
            noise_multiplier=self.noise_multiplier,
            num_microbatches=run_config["num-microbatches"],
            learning_rate=run_config["learning-rate"],
        )
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.losses.Reduction.NONE
        )
        self.model.compile(optimizer=self.optimizer, loss=loss, metrics=["accuracy"])

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)

        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=1,
            batch_size=self.batch_size,
        )

        compute_dp_sgd_privacy_statement(
            number_of_examples=self.x_train.shape[0],
            batch_size=self.batch_size,
            num_epochs=1,
            noise_multiplier=self.noise_multiplier,
            delta=1e-5,
        )

        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.compile(
            optimizer=self.optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {"accuracy": accuracy}


def client_fn(context: Context):
    model = Net()
    model.build(input_shape=(None, 28, 28, 1))

    partition_id = context.node_config["partition-id"]
    run_config = context.run_config
    noise_multiplier = 1.0 if partition_id % 2 == 0 else 1.5

    train_data, test_data = load_data(
        partition_id=partition_id,
        num_partitions=context.node_config["num-partitions"],
        batch_size=context.run_config["batch-size"],
    )

    return FlowerClient(
        model, train_data, test_data, noise_multiplier, run_config
    ).to_client()


app = ClientApp(client_fn=client_fn)
