"""$project_name: A Flower / $framework_str app."""

import numpy as np
from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from $import_name.task import get_params, load_model

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    input_dim = context.run_config["input-dim"]

    # Load global model
    model = load_model((input_dim,))
    arrays = ArrayRecord(get_params(model))

    # Initialize FedAvg strategy
    strategy = FedAvg()

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    ndarrays = result.arrays.to_numpy_ndarrays()
    np.savez("final_model.npz", *ndarrays)
