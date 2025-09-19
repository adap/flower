"""Quantum Federated Learning Server with PennyLane and Flower."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from pennylane_example.task import QuantumNet, test, load_data

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the quantum federated learning ServerApp."""
    
    # Read run config
    fraction_fit: float = context.run_config.get("fraction-fit", 1.0)
    fraction_evaluate: float = context.run_config.get("fraction-evaluate", 1.0)
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    
    print(f"Server configuration:")
    print(f"  - Number of rounds: {num_rounds}")
    print(f"  - Fraction fit: {fraction_fit}")
    print(f"  - Fraction evaluate: {fraction_evaluate}")
    print(f"  - Learning rate: {lr}")
    
    # Load global quantum model
    global_model = QuantumNet(num_classes=10)
    arrays = ArrayRecord(global_model.state_dict())
    
    print(f"Initialized quantum neural network with {sum(p.numel() for p in global_model.parameters())} parameters")
    
    # Initialize FedAvg strategy
    strategy = FedAvg(
        fraction_train=fraction_fit,
        fraction_evaluate=fraction_evaluate,
    )
    
    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )
    
    # Save final model to disk
    print("\nSaving final quantum model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_quantum_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate global quantum model on centralized test data."""
    
    # Load the model and initialize it with the received weights
    model = QuantumNet(num_classes=10)
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load centralized test data (using partition 0 as test set)
    _, testloader = load_data(partition_id=0, num_partitions=1, batch_size=128)
    
    # Evaluate the global model on the test set
    test_loss, test_accuracy = test(model, testloader, device)
    
    print(f"Global evaluation round {server_round}: Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")
    
    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_accuracy, "loss": test_loss})
