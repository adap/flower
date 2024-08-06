"""fl_dp_sa: A Flower / PyTorch app."""

from typing import List, Tuple

from flwr.server import Driver, LegacyContext, ServerApp, ServerConfig
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server.strategy import (
    DifferentialPrivacyClientSideFixedClipping,
    FedAvg,
)
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow

from fl_dp_sa.task import Net, get_weights


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
strategy = FedAvg(
    fraction_fit=0.2,
    fraction_evaluate=0.0,  # Disable evaluation for demo purpose
    min_fit_clients=20,
    min_available_clients=20,
    fit_metrics_aggregation_fn=weighted_average,
    initial_parameters=parameters,
)
strategy = DifferentialPrivacyClientSideFixedClipping(
    strategy, noise_multiplier=0.2, clipping_norm=10, num_sampled_clients=20
)


app = ServerApp()


@app.main()
def main(driver: Driver, context: Context) -> None:
    # Construct the LegacyContext
    context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=3),
        strategy=strategy,
    )

    # Create the train/evaluate workflow
    workflow = DefaultWorkflow(
        fit_workflow=SecAggPlusWorkflow(
            num_shares=7,
            reconstruction_threshold=4,
        )
    )

    # Execute
    workflow(driver, context)
