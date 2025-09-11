"""xgboost_quickstart: A Flower / XGBoost app."""

import numpy as np
import xgboost as xgb

from flwr.common import ArrayRecord, Context
from flwr.common.config import unflatten_dict
from flwr.server import Grid, ServerApp
from flwr.serverapp.strategy import FedXgbBagging

from xgboost_quickstart.task import replace_keys


# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    # Read run config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_train = context.run_config["fraction-train"]
    fraction_evaluate = context.run_config["fraction-evaluate"]
    # Flatted config dict and replace "-" with "_"
    cfg = replace_keys(unflatten_dict(context.run_config))
    params = cfg["params"]

    # Init global model
    global_model = b""  # Init with an empty object; the XGBooster will be created and trained on the client side.
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
    bst = xgb.Booster(params=params)
    global_model = bytearray(result.arrays["0"].numpy().tobytes())

    # Load global model into booster
    bst.load_model(global_model)

    # Save model
    print("\nSaving final model to disk...")
    bst.save_model("model.json")
