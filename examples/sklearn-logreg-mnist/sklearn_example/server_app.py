"""sklearn_example: A Flower / scikit-learn app."""

from typing import Dict

# import numpy as np
from flwr_datasets import FederatedDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn_example.task import set_initial_params, set_model_params

from flwr.common import Context, NDArrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg


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


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour."""

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Define the strategy
    strategy = FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(),
        on_fit_config_fn=fit_round,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
