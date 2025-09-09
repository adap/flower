"""xgboost_quickstart: A Flower / XGBoost app."""

import numpy as np

from flwr.common import ArrayRecord, Context
from flwr.server import Grid, ServerApp
from flwr.serverapp.strategy import FedXgbBagging


# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    # Read run config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_train = context.run_config["fraction-train"]
    fraction_evaluate = context.run_config["fraction-evaluate"]

    # Init global model
    global_model = b""  # Init with empty object
    arrays = ArrayRecord([np.frombuffer(global_model, dtype=np.uint8)])

    # Initialize FedXgbBagging strategy
    strategy = FedXgbBagging(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
    )

    # Start strategy, run FedXgbBagging for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    model_arr = bytearray(result.arrays['0'].numpy().tobytes())
    with open("final_model.json", "wb") as f:
        f.write(model_arr)
