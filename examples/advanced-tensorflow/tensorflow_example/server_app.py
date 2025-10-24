"""tensorflow-example: A Flower / TensorFlow app."""

from datasets import load_dataset
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from tensorflow_example.strategy import CustomFedAvg
from tensorflow_example.task import create_run_dir, load_model

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    # Read run config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_train = context.run_config["fraction-train"]
    fraction_eval = context.run_config["fraction-evaluate"]

    # Load global model
    global_model = load_model()
    arrays = ArrayRecord(global_model.get_weights())

    # Initialize FedAvg strategy
    strategy = CustomFedAvg(
        fraction_train=fraction_train, fraction_evaluate=fraction_eval
    )

    # Define directory for results and save config
    save_path, run_dir = create_run_dir(config=context.run_config)
    strategy.set_save_path_and_run_dir(save_path, run_dir)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": 0.001}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    global_model.set_weights(result.arrays.to_numpy_ndarrays())
    global_model.save("final_model.keras")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # This is the exact same dataset as the one downloaded by the clients via
    # FlowerDatasets. However, we don't use FlowerDatasets for the server since
    # partitioning is not needed.
    # We make use of the "test" split only
    global_test_set = load_dataset("zalando-datasets/fashion_mnist")["test"]
    global_test_set.set_format("numpy")

    x_test, y_test = global_test_set["image"] / 255.0, global_test_set["label"]

    net = load_model()
    net.set_weights(arrays.to_numpy_ndarrays())
    loss, accuracy = net.evaluate(x_test, y_test, verbose=0)
    return MetricRecord({"accuracy": accuracy, "loss": loss})
