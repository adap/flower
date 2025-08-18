"""app-pytorch: A Flower / PyTorch app."""

from app_pytorch.task import Net
from flwr.common import ArrayRecord, ConfigRecord, Context, RecordDict
from flwr.common.logger import log
from flwr.server import Grid, ServerApp

from .fedavg import FedAvg

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:

    # Init global model
    global_model = Net()

    # Init strategy
    strategy = FedAvg(fraction_train=context.run_config["fraction-train"])

    # Prepare payload to communicate
    recorddict = RecordDict(
        {
            "global-model": ArrayRecord(global_model.state_dict()),
            "clientapp-train-config": ConfigRecord({"lr": 0.01}),
            "clientapp-evaluate-config": ConfigRecord(
                {"num-batches": 10, "save-model-checkpoint": True}
            ),
        }
    )

    # Execute strategy loop
    num_rounds = context.run_config["num-server-rounds"]
    metrics = strategy.run(recorddict, grid, num_rounds=num_rounds, timeout=3600)

    print(metrics)
