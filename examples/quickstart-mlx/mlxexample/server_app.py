"""mlxexample: A Flower / MLX app."""

from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from mlxexample.task import MLP, get_params


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate custom `accuracy` metric by weighted average."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour."""

    # Init model
    model = MLP(
        num_layers=context.run_config["num-layers"],
        input_dim=context.run_config["img-size"] ** 2,
        hidden_dim=context.run_config["hidden-dim"],
    )

    # Convert model parameters to flwr.common.Parameters
    ndarrays = get_params(model)
    global_model_init = ndarrays_to_parameters(ndarrays)

    # Define the strategy
    fraction_eval = context.run_config["fraction-evaluate"]
    strategy = FedAvg(
        fraction_evaluate=fraction_eval,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=global_model_init,
    )

    # Construct ServerConfig
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
