"""sklearn_example: A Flower / scikit-learn app."""

from typing import Dict

import numpy as np
from flwr_datasets import FederatedDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from flwr.common import NDArrays
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg


def set_model_params(model: LogisticRegression, params: NDArrays) -> LogisticRegression:
    """Sets the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: LogisticRegression):
    """Sets initial parameters as zeros Required since model params are uninitialized
    until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer to
    sklearn.linear_model.LogisticRegression documentation for more information.
    """
    n_classes = 10  # MNIST has 10 classes
    n_features = 784  # Number of features in dataset
    model.classes_ = np.array([i for i in range(10)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))


def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn():
    """Return an evaluation function for server-side evaluation."""

    model = LogisticRegression()
    set_initial_params(model)

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    fds = FederatedDataset(dataset="mnist", partitioners={"train": 10})
    dataset = fds.load_split("test").with_format("numpy")
    X_test, y_test = dataset["image"].reshape((len(dataset), -1)), dataset["label"]

    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: NDArrays, config):
        # Update model with the latest parameters
        set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate


strategy = FedAvg(
    min_available_clients=2,
    evaluate_fn=get_evaluate_fn(),
    on_fit_config_fn=fit_round,
)

# Config Flower server for five rounds of federated learning
config = ServerConfig(num_rounds=5)

app = ServerApp(
    config=config,
    strategy=strategy,
)
