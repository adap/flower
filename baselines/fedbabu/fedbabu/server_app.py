"""FedBABU Server Application.

This module implements the Federated Learning server for the FedBABU (Federated Learning
with Body and Head Update) approach, as described in the paper "FedBABU: Towards Enhanced Representation for Federated Image Classification".

Key Features:
- Coordinates federated learning across multiple clients
- Implements parameter aggregation using Federated Averaging (FedAvg)
- Aggregates only feature extractor parameters while preserving local classifiers
- Provides weighted metrics aggregation for evaluation
- Supports configurable client participation rates

The server process follows these steps:
1. Initializes global model parameters
2. Selects a fraction of available clients for training
3. Aggregates feature extractor updates from clients
4. Evaluates global model performance using weighted metrics
5. Manages multiple rounds of federation
"""

from typing import List, Tuple

from fedbabu.task import FourConvNet, get_weights
from flwr.common import Context, Metrics, Parameters, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

# Server configuration constants
MIN_AVAILABLE_CLIENTS = 2  # Minimum number of clients required for federation
EVALUATE_FRACTION = 1.0  # Fraction of clients to use for evaluation (1.0 = all clients)


def get_initial_parameters() -> Parameters:
    """Initialize model parameters for federated learning.

    This function creates the initial global model state by:
    1. Instantiating a fresh FourConvNet model
    2. Extracting its parameters
    3. Converting parameters to Flower's format

    The initialized parameters serve as the starting point for federated training,
    ensuring all clients begin from the same model state.

    Returns:
        Parameters: Initial model parameters ready for distribution to clients
    """
    # Create a new model instance and get its parameters
    model = FourConvNet()
    ndarrays = get_weights(model)
    return ndarrays_to_parameters(ndarrays)


def evaluate_metrics_aggregation_fn(eval_metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate evaluation metrics across clients using weighted averaging.

    This function combines metrics from multiple clients by:
    1. Extracting sample counts and metric values from each client
    2. Computing weighted averages based on number of samples
    3. Producing aggregate accuracy and loss metrics

    The weighting ensures that clients with more samples have proportionally
    more influence on the final metrics.

    Args:
        eval_metrics (List[Tuple[int, Metrics]]): List of tuples containing:
            - Number of examples used for evaluation
            - Dictionary of metrics (accuracy and loss) from each client

    Returns:
        Metrics: Dictionary containing aggregated metrics:
            - loss: Weighted average of client losses
            - accuracy: Weighted average of client accuracies
    """
    weights, accuracies, losses = [], [], []
    for num_examples, metric in eval_metrics:
        weights.append(num_examples)
        accuracies.append(float(metric["accuracy"]) * num_examples)
        losses.append(float(metric["loss"]) * num_examples)
    accuracy = sum(accuracies) / sum(weights)
    loss = sum(losses) / sum(weights)
    return {"loss": loss, "accuracy": accuracy}


def server_fn(context: Context) -> ServerAppComponents:
    """Create and configure a Flower server for FedBABU.

    This factory function sets up the federated learning server by:
    1. Extracting configuration from the provided context
    2. Initializing global model parameters
    3. Configuring the FedAvg strategy with appropriate settings
    4. Setting up server configuration for training rounds

    The server implements the FedBABU strategy where only feature extractor
    parameters are aggregated while classifier parameters remain local to
    each client, promoting better generalization.

    Args:
        context (Context): Server context containing configuration including:
            - run_config: Federation settings including:
                * num-server-rounds: Total number of training rounds
                * fraction-fit: Fraction of clients to select for training

    Returns:
        ServerAppComponents: Configured server containing:
            - strategy: FedAvg strategy with custom metric aggregation
            - config: Server configuration with number of rounds
    """
    # Extract configuration parameters
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Get initial model parameters
    initial_parameters = get_initial_parameters()

    def on_fit_and_evaluate_config_fn(server_round: int) -> dict:
        """Configuration function for client training and evaluation.

        Args:
            server_round: Current round number of the server

        Returns:
            dict: Configuration dictionary containing:
                - lr: Learning rate for the current round, adjusted at specific intervals
        """
        lr = context.run_config["lr"]
        if (server_round + 1) in [num_rounds // 2, (num_rounds * 3) // 4]:
            lr *= 0.1
        return {"lr": lr}

    # Configure federated averaging strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=EVALUATE_FRACTION,
        min_available_clients=MIN_AVAILABLE_CLIENTS,
        initial_parameters=initial_parameters,
        on_fit_config_fn=on_fit_and_evaluate_config_fn,
        on_evaluate_config_fn=on_fit_and_evaluate_config_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )

    # Configure server
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create and configure the Flower server application
app = ServerApp(server_fn=server_fn)
