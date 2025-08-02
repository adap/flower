"""jaxexample: A Flower / JAX app."""

from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from flax import nnx
from jaxexample.task import get_params, CNN


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Initialize global model
    rng = nnx.Rngs(0)
    model = CNN(rngs=rng)
    graphdef, params, _ = nnx.split(model, nnx.Param, ...)
    params = get_params(params)
    initial_parameters = ndarrays_to_parameters(params)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=0.4,
        fraction_evaluate=0.5,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=initial_parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
