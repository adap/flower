"""app-pytorch: A Flower / PyTorch app."""

from app_pytorch.task import Net

from flwr.common import (
    ArrayRecord,
    Context,
    RecordDict,
    ConfigRecord,
)
from flwr.common.logger import log
from flwr.server import Grid, ServerApp
from .fedavg import FedAvgA, run_strategy

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:

    num_rounds = context.run_config["num-server-rounds"]

        # Init global model
    global_model = Net()
    global_model_key = "model"

    strategy = FedAvgA()

    gmodel_record = ArrayRecord(global_model.state_dict())
    recorddict = RecordDict(
        {
            global_model_key: gmodel_record,
            "train-config": ConfigRecord({"lr": 0.01}),
        }
    )

    run_strategy(recorddict, strategy, grid, num_rounds=num_rounds, timeout=3600)
