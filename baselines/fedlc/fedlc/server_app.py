"""fedlc: A Flower Baseline."""

from typing import List, Tuple

from flwr.common import Context, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from .strategy import CheckpointedFedAvg, CheckpointedFedProx
from flwr.common.logger import log
from logging import INFO, DEBUG
from fedlc.model import initialize_model

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Do weighted average of accuracy metric."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * float(m["accuracy"]) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    weighted_accuracy = sum(accuracies) / sum(examples)
    log(INFO, f"Weighted average accuracy: {weighted_accuracy}")
    return {"accuracy": weighted_accuracy}


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_evaluate = context.run_config["fraction-evaluate"]
    proximal_mu = context.run_config["proximal-mu"]
    save_params_every = context.run_config["save-params-every"]
    log(DEBUG, f"Saving params every {save_params_every} rounds")

    num_classes = int(context.run_config["num-classes"])
    num_channels = int(context.run_config["num-channels"])
    model_name = str(context.run_config["model-name"])

    net = initialize_model(model_name, num_channels, num_classes)

    # Define strategy
    if context.run_config["strategy"] == "fedprox":
        strategy = CheckpointedFedProx(
            net=net,
            run_config=context.run_config,
            fraction_fit=float(fraction_fit),
            fraction_evaluate=float(fraction_evaluate),
            evaluate_metrics_aggregation_fn=weighted_average,
            proximal_mu=float(proximal_mu),
        )
    else:
        # default to FedAvg
        strategy = CheckpointedFedAvg(
            net=net, 
            run_config=context.run_config,
            fraction_fit=float(fraction_fit),
            fraction_evaluate=float(fraction_evaluate),
            evaluate_metrics_aggregation_fn=weighted_average,
        )
    config = ServerConfig(num_rounds=int(num_rounds))

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)