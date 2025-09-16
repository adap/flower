"""tfexample: A Flower / TensorFlow app."""

from typing import List, Tuple

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from tfexample.task import load_model

# Create the ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp"""
    # Load config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_train = context.run_config["fraction-train"]

    # Load initial model
    initial_model = load_model()
    arrays = ArrayRecord(initial_model.get_weights())

    # Define and start FedAvg strategy
    strategy = FedAvg(
        fraction_train=fraction_train,
    )

    result = strategy.start(
        grid=grid,
        initial_arrays = arrays,
        num_rounds=num_rounds,
    )
    print(result)
 
