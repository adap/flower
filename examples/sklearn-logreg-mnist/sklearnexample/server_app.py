"""sklearnexample: A Flower / scikit-learn app."""

from flwr.common import Context, NDArrays, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr_datasets import FederatedDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from sklearnexample.task import (
    create_log_reg_and_instantiate_parameters,
    get_model_parameters,
    set_initial_params,
    set_model_params,
)


def get_evaluate_fn(penalty):
    """Return an evaluation function for server-side evaluation."""

    model = LogisticRegression(penalty=penalty)
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

    penalty = context.run_config["penalty"]
    model = create_log_reg_and_instantiate_parameters(penalty)
    ndarrays = get_model_parameters(model)
    global_model_init = ndarrays_to_parameters(ndarrays)

    # Define the strategy
    fraction_fit = context.run_config["fraction-fit"]
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        evaluate_fn=get_evaluate_fn(penalty),
        initial_parameters=global_model_init,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
