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

    num_rounds = context.run_config["num-server-rounds"]

    # Init global model
    global_model = Net()
    # Init strategy
    strategy = FedAvg()

    # Prepare payload to communicate
    #! We could be passing the `clientapp-...-config` when constructing the strategy
    #! However, since those will be communicated in a Message, it might be better to
    #! keep them as part of a single RecordDict along with the model
    recorddict = RecordDict(
        {
            "global-model": ArrayRecord(global_model.state_dict()),
            "clientapp-train-config": ConfigRecord({"lr": 0.01}),
            "clientapp-evaluate-config": ConfigRecord(
                {"num-batches": 10, "save-model-checkpoint": True}
            ),
        }
    )

    metrics = strategy.run(recorddict, grid, num_rounds=num_rounds, timeout=3600)
