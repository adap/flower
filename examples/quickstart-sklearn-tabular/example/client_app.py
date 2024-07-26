import argparse
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from flwr.common import Context
from flwr.client import NumPyClient, ClientApp

from example.task import (
    set_model_params,
    get_model_parameters,
    set_initial_params,
    load_data,
    UNIQUE_LABELS,
)


class FlowerClient(NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.unique_labels = UNIQUE_LABELS

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        set_model_params(self.model, parameters)
        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)
        accuracy = self.model.score(self.X_train, self.y_train)
        return (
            get_model_parameters(self.model),
            len(self.X_train),
            {"train_accuracy": accuracy},
        )

    def evaluate(self, parameters, config):  # type: ignore
        set_model_params(self.model, parameters)
        y_pred = self.model.predict_proba(self.X_test)
        loss = log_loss(self.y_test, y_pred, labels=self.unique_labels)
        accuracy = self.model.score(self.X_test, self.y_test)
        return loss, len(self.X_test), {"test_accuracy": accuracy}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp.

    You can use settings in `context.run_config` to parameterize the
    construction of your Client. You could use the `context.node_config` to, for
    example, indicate which dataset to load (e.g., accessing the partition-id).
    """

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    X_train, y_train, X_test, y_test = load_data(partition_id, num_partitions)

    # Read the run config to get settings to configure the Client
    penalty = context.run_config["penalty"]

    # Create LogisticRegression Model
    model = LogisticRegression(
        penalty=penalty,
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )

    # Setting initial parameters, akin to model.compile for keras models
    set_initial_params(model, n_features=X_train.shape[1], n_classes=len(UNIQUE_LABELS))

    # Return Client instance
    return FlowerClient(model, X_train, y_train, X_test, y_test).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)
