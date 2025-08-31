"""$project_name: A Flower / $framework_str app."""

from pprint import pprint

from flwr.common import ArrayRecord, Context
from flwr.server import Grid, ServerApp
from flwr.serverapp import FedAvg

from $import_name.task import load_model

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    num_rounds: int = context.run_config["num-server-rounds"]

    # Load global model
    global_model = load_model()
    arrays = ArrayRecord(global_model.get_weights())

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_train=1.0, fraction_evaluate=1.0)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    # Log resulting metrics
    print("\nDistributed train metrics:")
    pprint(result.train_metrics_clientapp)
    print("\nDistributed evaluate metrics:")
    pprint(result.evaluate_metrics_clientapp)

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
