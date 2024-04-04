from typing import List, Tuple

from task import Net, get_weights

import flwr as fl
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import Driver, LegacyContext


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    examples = [num_examples for num_examples, _ in metrics]

    # Multiply accuracy of each client by number of examples used
    train_losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    train_accuracies = [
        num_examples * m["train_accuracy"] for num_examples, m in metrics
    ]
    val_losses = [num_examples * m["val_loss"] for num_examples, m in metrics]
    val_accuracies = [num_examples * m["val_accuracy"] for num_examples, m in metrics]

    # Aggregate and return custom metric (weighted average)
    return {
        "train_loss": sum(train_losses) / sum(examples),
        "train_accuracy": sum(train_accuracies) / sum(examples),
        "val_loss": sum(val_losses) / sum(examples),
        "val_accuracy": sum(val_accuracies) / sum(examples),
    }


# Initialize model parameters
ndarrays = get_weights(Net())
parameters = ndarrays_to_parameters(ndarrays)


# Define strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Select all available clients
    fraction_evaluate=0.0,  # Disable evaluation
    min_available_clients=2,
    fit_metrics_aggregation_fn=weighted_average,
    initial_parameters=parameters,
)


# Run via `flower-server-app server_workflow:app`
app = fl.server.ServerApp()


@app.main()
def main(driver: Driver, context: Context) -> None:
    # Construct the LegacyContext
    context = LegacyContext(
        state=context.state,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

    # Create the workflow
    workflow = fl.server.workflow.DefaultWorkflow()

    # Execute
    workflow(driver, context)
