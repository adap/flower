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


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class FlowerClient(NumPyClient):
    def __init__(
        self,
        model,
        train_data,
        test_data,
        l2_norm_clip,
        noise_multiplier,
        num_microbatches,
        learning_rate,
        batch_size,
    ) -> None:
        super().__init__()
        self.model = model
        self.x_train, self.y_train = train_data
        self.x_test, self.y_test = test_data
        self.noise_multiplier = noise_multiplier
        self.l2_norm_clip = l2_norm_clip
        self.num_microbatches = num_microbatches
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        if self.batch_size % self.num_microbatches != 0:
            raise ValueError(
                f"Batch size {self.batch_size} is not divisible by the number of microbatches {self.num_microbatches}"
            )

        self.optimizer = tensorflow_privacy.DPKerasSGDOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=num_microbatches,
            learning_rate=learning_rate,
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

    l2_norm_clip = 1.0
    num_microbatches = 64
    learning_rate = 0.01
    batch_size = 64
    noise_multiplier = 1.0 if partition_id % 2 == 0 else 1.5

    partition_id = context.node_config["partition-id"]
    train_data, test_data = load_data(
        partition_id=partition_id,
        num_partitions=context.node_config["num-partitions"],
        batch_size=batch_size,
    )

    return FlowerClient(
        model,
        train_data,
        test_data,
        noise_multiplier,
        l2_norm_clip,
        num_microbatches,
        learning_rate,
        batch_size,
    ).to_client()


app = ClientApp(client_fn=client_fn)
