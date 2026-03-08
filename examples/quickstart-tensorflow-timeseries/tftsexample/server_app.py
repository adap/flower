"""timeseries: A Flower / TensorFlow app."""

from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from tftsexample.task import load_model, load_centralized_dataset, test

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    num_rounds: int = context.run_config["num-server-rounds"]

    # Load global model
    model = load_model()
    arrays = ArrayRecord(model.get_weights())

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_train=1.0, fraction_evaluate=1.0)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    ndarrays = result.arrays.to_numpy_ndarrays()
    model.set_weights(ndarrays)
    model.save("final_model.keras")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""
    
    # Load the model and initialize it with the received weights
    model = load_model()
    ndarrays = arrays.to_numpy_ndarrays()
    model.set_weights(ndarrays)

    # Load entire test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model on the test set
    test_loss, test_acc = test(model, test_dataloader)

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})