"""Quantum Federated Learning Server with PennyLane and Flower."""

from typing import List, Tuple
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from pennylane_example.task import QuantumNet, get_weights


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate evaluation metrics using weighted average."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Return weighted average accuracy
    return {"accuracy": sum(accuracies) / sum(examples)}


def weighted_average_fit(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate fit metrics using weighted average."""
    # Extract train and validation metrics
    train_losses = [num_examples * m.get("train_loss", 0) for num_examples, m in metrics]
    val_losses = [num_examples * m.get("val_loss", 0) for num_examples, m in metrics]  
    val_accuracies = [num_examples * m.get("val_accuracy", 0) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    total_examples = sum(examples)
    
    aggregated_metrics = {}
    if total_examples > 0:
        aggregated_metrics = {
            "train_loss": sum(train_losses) / total_examples,
            "val_loss": sum(val_losses) / total_examples,
            "val_accuracy": sum(val_accuracies) / total_examples,
        }
    
    return aggregated_metrics


def server_fn(context: Context):
    """Construct components for the quantum federated learning server."""
    
    # Read configuration from context
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config.get("fraction-fit", 1.0)
    fraction_evaluate = context.run_config.get("fraction-evaluate", 1.0)
    min_available_clients = context.run_config.get("min-available-clients", 2)
    
    print(f"Server configuration:")
    print(f"  - Number of rounds: {num_rounds}")
    print(f"  - Fraction fit: {fraction_fit}")
    print(f"  - Fraction evaluate: {fraction_evaluate}")
    print(f"  - Min available clients: {min_available_clients}")
    
    # Initialize quantum neural network to get initial parameters
    net = QuantumNet(num_classes=10)
    ndarrays = get_weights(net)
    parameters = ndarrays_to_parameters(ndarrays)
    
    print(f"Initialized quantum neural network with {sum(p.numel() for p in net.parameters())} parameters")
    
    # Define federated learning strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_available_clients,
        min_evaluate_clients=min_available_clients,
        min_available_clients=min_available_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average_fit,
        initial_parameters=parameters,
    )
    
    # Configure server
    config = ServerConfig(num_rounds=num_rounds)
    
    return ServerAppComponents(strategy=strategy, config=config)


# Create Flower ServerApp
app = ServerApp(server_fn=server_fn)
