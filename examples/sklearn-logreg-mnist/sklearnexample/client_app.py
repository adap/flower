"""sklearnexample: A Flower / scikit-learn app."""

import warnings

from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context
from sklearn.metrics import log_loss

from sklearnexample.task import (
    create_log_reg_and_instantiate_parameters,
    get_model_parameters,
    load_data,
    set_model_params,
)


# Define Flower client
class MnistClient(NumPyClient):
    def __init__(self, model, X_train, X_test, y_train, y_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def fit(self, parameters, config):
        set_model_params(self.model, parameters)
        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)
        return get_model_parameters(self.model), len(self.X_train), {}

    def evaluate(self, parameters, config):
        set_model_params(self.model, parameters)
        loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
        accuracy = self.model.score(self.X_test, self.y_test)
        return loss, len(self.X_test), {"accuracy": accuracy}


def client_fn(context: Context) -> Client:
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    X_train, X_test, y_train, y_test = load_data(partition_id, num_partitions)

    # Read the run config to get settings to configure the Client
    penalty = context.run_config["penalty"]

    # Create LogisticRegression Model
    model = create_log_reg_and_instantiate_parameters(penalty)

    # Return Client instance
    return MnistClient(model, X_train, X_test, y_train, y_test).to_client()


# Create ClientApp
app = ClientApp(client_fn=client_fn)
