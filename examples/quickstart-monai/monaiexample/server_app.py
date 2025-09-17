"""monaiexample: A Flower / MONAI app."""

from typing import List, Tuple

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from monaiexample.task import get_params, load_model

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]

    # Load global model
    model = load_model()
    arrays = ArrayRecord(model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_train=fraction_train)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    # Save final model to disk
    final_model_name = "final_model.pt"
    print(f"\nSaving {final_model_name} to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, final_model_name)
