"""$project_name: A Flower / $framework_str app."""

import numpy as np
from flwr.common import ArrayRecord, Context
from flwr.server import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from $import_name.task import get_dummy_model

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    num_rounds: int = context.run_config["num-server-rounds"]

    # Load global model
    global_model = get_dummy_model()
    arrays = ArrayRecord(global_model)

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
    np.savez("final_model", *ndarrays)
