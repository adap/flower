import argparse
import os

import tensorflow as tf
import tensorflow_privacy
from flwr.client import ClientApp, NumPyClient
from flwr_datasets import FederatedDataset
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import (
    compute_dp_sgd_privacy_statement,
)

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_data(partition_id, batch_size):
    fds = FederatedDataset(dataset="mnist", partitioners={"train": 2})
    partition = fds.load_partition(partition_id, "train")
    partition.set_format("numpy")

    # Divide data on each node: 80% train, 20% test
    partition = partition.train_test_split(test_size=0.2, seed=42)
    x_train, y_train = partition["train"]["image"] / 255.0, partition["train"]["label"]
    x_test, y_test = partition["test"]["image"] / 255.0, partition["test"]["label"]

    # Adjust the size of the training dataset to make it evenly divisible by the batch size
    remainder = len(x_train) % batch_size
    if remainder != 0:
        x_train = x_train[:-remainder]
        y_train = y_train[:-remainder]

    return (x_train, y_train), (x_test, y_test)


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


def client_fn_parameterized(
    partition_id,
    noise_multiplier,
    l2_norm_clip=1.0,
    num_microbatches=64,
    learning_rate=0.01,
    batch_size=64,
):
    def client_fn(cid: str):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )
        train_data, test_data = load_data(
            partition_id=partition_id, batch_size=batch_size
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

    return client_fn


appA = ClientApp(
    client_fn=client_fn_parameterized(partition_id=0, noise_multiplier=1.0),
)

appB = ClientApp(
    client_fn=client_fn_parameterized(partition_id=1, noise_multiplier=1.5),
)
