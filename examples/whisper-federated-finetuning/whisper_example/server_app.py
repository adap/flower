"""whisper_example: A Flower / PyTorch app with OpenAi's Whisper."""

from typing import List, Tuple

from whisper_example.model import get_model, get_params

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    num_classes = context.run_config["num-classes"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize global model parameters. Recall we are
    # only federating the classification head
    _, classifier = get_model("cpu", num_classes, False)
    ndarrays = get_params(classifier)
    parameters = ndarrays_to_parameters(ndarrays)

    # Define the strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=0.0,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
