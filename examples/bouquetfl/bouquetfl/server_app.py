"""bouquetfl: A Flower / PyTorch app."""

import os
import time

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.common import Context
from flwr.server import ServerApp
from flwr.server.strategy import FedAvg
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from bouquetfl import task
from bouquetfl.utils import formatting
from bouquetfl.utils.sampler import generate_hardware_config
import subprocess


# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # IMPORTANT: Before setup, set client resources in ./flwr/config.toml to maximum local hardware.
    # Example: options.backend.client-resources.num-cpus = 8
    #          options.backend.client-resources.num-gpus = 1

    mps_proc = subprocess.Popen(["nvidia-cuda-mps-control", "-d"])
    mps_proc.wait()

    # Generate hardware profiles for clients, if not predefined in YAML file
    if not os.path.exists("./bouquetfl/config/federation_client_hardware.yaml"):
        generate_hardware_config(num_clients=context.run_config["num-partitions"])

    arrays = ArrayRecord(task.get_initial_state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_evaluate=context.run_config["fraction-evaluate"])

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": context.run_config["learning-rate"]}),
        num_rounds=context.run_config["num-server-rounds"],
        evaluate_fn=global_evaluate,
    )

    # Save final model to disk
    torch.save(result.global_arrays.to_torch_state_dict(), "./bouquetfl/checkpoints/final_model.pt")

def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = task.get_model()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = task.load_global_test_data()

    # Evaluate the global model on the test set
    test_loss, test_acc = task.test(model, test_dataloader, device)
    formatting.print_timings()
    time.sleep(0.5)
    # Return the evaluation metrics
    return MetricRecord({"loss": test_loss, "accuracy": test_acc})
