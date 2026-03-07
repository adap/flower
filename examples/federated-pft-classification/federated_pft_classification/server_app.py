"""xgboost_comprehensive: A Flower / XGBoost app."""

import numpy as np
import xgboost as xgb
from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.common.config import unflatten_dict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedXgbBagging, FedXgbCyclic

from federated_pft_classification.task import replace_keys

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_train = context.run_config["fraction-train"]
    fraction_evaluate = context.run_config["fraction-evaluate"]
    train_method = context.run_config["train-method"]

    # Flatten config dict and replace "-" with "_"
    cfg = replace_keys(unflatten_dict(context.run_config))
    params = cfg["params"]

    # Init global model
    global_model = b""  # Init with an empty object; the XGBooster will be created and trained on the client side.
    arrays = ArrayRecord([np.frombuffer(global_model, dtype=np.uint8)])

    # Define strategy
    if train_method == "bagging":
        strategy = FedXgbBagging(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
        )
    else:
        strategy = FedXgbCyclic()

    # Start strategy, run for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
        evaluate_fn=None,
    )

    # Save final model to disk
    bst = xgb.Booster(params=params)
    global_model = bytearray(result.arrays["0"].numpy().tobytes())
    bst.load_model(global_model)

    print("\nSaving final model to disk...")
    bst.save_model("final_model.json")
