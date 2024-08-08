"""sklearn_example: A Flower / scikit-learn app."""

import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn_example.task import (
    get_model_parameters,
    load_data,
    set_initial_params,
    set_model_params,
)

from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context


# Define Flower client
class MnistClient(NumPyClient):
    def __init__(
        self, model, X_train, X_test, y_train, y_test
    ):  # pylint: disable=R0913
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def get_parameters(self, config):  # type: ignore
        return get_model_parameters(self.model)

    def fit(self, parameters, config):  # type: ignore
        set_model_params(self.model, parameters)
        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)
        print(f"Training finished for round {config['server_round']}")
        return get_model_parameters(self.model), len(self.X_train), {}

    def evaluate(self, parameters, config):  # type: ignore
        set_model_params(self.model, parameters)
        loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
        accuracy = self.model.score(self.X_test, self.y_test)
        return loss, len(self.X_test), {"accuracy": accuracy}


def client_fn(context: Context) -> Client:
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Load train and test data
    X_train, X_test, y_train, y_test = load_data(partition_id, num_partitions)

    # Create LogisticRegression Model
    model = LogisticRegression(
        penalty="l2",
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )

    # Set initial parameters, akin to model.compile for keras models
    set_initial_params(model)

    # Return Client instance
    return MnistClient(model, X_train, X_test, y_train, y_test).to_client()


# Create ClientApp
app = ClientApp(client_fn=client_fn)
