"""tf_privacy: Training with Sample-Level Differential Privacy using TensorFlow-Privacy Engine."""

import os

import numpy as np
import tensorflow as tf
import tensorflow_privacy
from flwr.client import NumPyClient
from flwr.clientapp import ClientApp
from flwr.common import Context
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import (
    compute_dp_sgd_privacy_statement,
)

from tf_privacy.task import load_data, load_model

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class FlowerClient(NumPyClient):
    def __init__(
        self,
        train_data,
        test_data,
        noise_multiplier,
        run_config,
    ) -> None:
        super().__init__()
        self.model = load_model()
        self.x_train, self.y_train = train_data
        self.x_train = np.expand_dims(self.x_train, axis=-1)
        self.x_test, self.y_test = test_data
        self.x_test = np.expand_dims(self.x_test, axis=-1)
        self.noise_multiplier = noise_multiplier
        self.run_config = run_config
        if self.run_config["batch-size"] % self.run_config["num-microbatches"] != 0:
            raise ValueError(
                f"Batch size {self.run_config['batch-size']} is not divisible by the number of microbatches {self.run_config['num-microbatches']}"
            )

        self.optimizer = tensorflow_privacy.DPKerasSGDOptimizer(
            l2_norm_clip=self.run_config["l2-norm-clip"],
            noise_multiplier=self.noise_multiplier,
            num_microbatches=self.run_config["num-microbatches"],
            learning_rate=self.run_config["learning-rate"],
        )
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.losses.Reduction.NONE
        )
        self.model.compile(optimizer=self.optimizer, loss=loss, metrics=["accuracy"])

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=1,
            batch_size=self.run_config["batch-size"],
        )

        dp_statement = compute_dp_sgd_privacy_statement(
            number_of_examples=self.x_train.shape[0],
            batch_size=self.run_config["batch-size"],
            num_epochs=1,
            noise_multiplier=self.noise_multiplier,
            delta=1e-5,
        )
        print(dp_statement)

        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {"accuracy": accuracy}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    run_config = context.run_config
    noise_multiplier = 1.0 if partition_id % 2 == 0 else 1.5

    train_data, test_data = load_data(
        partition_id=partition_id,
        num_partitions=context.node_config["num-partitions"],
        batch_size=context.run_config["batch-size"],
    )

    return FlowerClient(train_data, test_data, noise_multiplier, run_config).to_client()


app = ClientApp(client_fn=client_fn)
