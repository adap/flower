"""jaxexample: A Flower / JAX app."""

import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from jaxexample.task import (
    apply_model,
    create_train_state,
    get_params,
    load_data,
    set_params,
    train,
)


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, train_state, trainset, testset):
        self.train_state = train_state
        self.trainset, self.testset = trainset, testset

    def fit(self, parameters, config):
        self.train_state = set_params(self.train_state, parameters)
        self.train_state, loss, acc = train(self.train_state, self.trainset)
        params = get_params(self.train_state.params)
        return (
            params,
            len(self.trainset),
            {"train_acc": float(acc), "train_loss": float(loss)},
        )

    def evaluate(self, parameters, config):
        self.train_state = set_params(self.train_state, parameters)

        losses = []
        accs = []
        for batch in self.testset:
            _, loss, accuracy = apply_model(
                self.train_state, batch["image"], batch["label"]
            )
            losses.append(float(loss))
            accs.append(float(accuracy))

        return np.mean(losses), len(self.testset), {"accuracy": np.mean(accs)}


def client_fn(context: Context):

    num_partitions = context.node_config["num-partitions"]
    partition_id = context.node_config["partition-id"]
    batch_size = context.run_config["batch-size"]
    trainset, testset = load_data(partition_id, num_partitions, batch_size)

    # Create train state object (model + optimizer)
    lr = context.run_config["learning-rate"]
    train_state = create_train_state(lr)

    # Return Client instance
    return FlowerClient(train_state, trainset, testset).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
