"""bouquetfl: A Flower / PyTorch app."""

import json
import os
import tomllib

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.common import Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from bouquetfl.utils.sampler import generate_hardware_config
from bouquetfl.utils.localinfo import get_all_local_info

app = ServerApp()

HARDWARE_CONFIG_PATH = "config/federation_client_hardware.toml"


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    run_config      = context.run_config
    num_rounds:  int   = run_config["num-server-rounds"]
    num_clients: int   = run_config["num-clients"]
    lr:          float = run_config["learning-rate"]
    experiment:  str   = run_config["experiment"]

    if experiment == "cifar10":
        from task import cifar10 as flower_baseline
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    # Profile local hardware once — passed to clients via train_config
    local_hw = get_all_local_info()

    # Load or generate hardware profiles for all clients
    if os.path.exists(HARDWARE_CONFIG_PATH):
        with open(HARDWARE_CONFIG_PATH, "rb") as f:
            hardware_config = tomllib.load(f)
        print(f"[server] loaded hardware config from {HARDWARE_CONFIG_PATH}")
    else:
        hardware_config = generate_hardware_config(num_clients=num_clients, local_hw=local_hw)
        print("[server] generated hardware config from local hardware profile")

    # Print full federation hardware config for traceability
    print("[server] federation hardware config:")
    for client_id, profile in hardware_config.items():
        print(f"  {client_id}: GPU={profile['gpu']}  CPU={profile['cpu']}  RAM={profile['ram_gb']} GB")

    arrays   = ArrayRecord(flower_baseline.get_initial_state_dict())
    strategy = FedAvg(
        fraction_train=run_config["fraction-fit"],
        fraction_evaluate=run_config["fraction-evaluate"],
    )

    strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({
            "hardware_config": json.dumps(hardware_config),
            "local_hw":        json.dumps(local_hw),
            "lr":              lr,
        }),
        num_rounds=num_rounds,
        evaluate_fn=lambda server_round, arrays: global_evaluate(
            server_round, arrays, flower_baseline
        ),
    )

def global_evaluate(server_round: int, arrays: ArrayRecord, flower_baseline) -> MetricRecord:
    """Evaluate global model on central test data."""

    model = flower_baseline.get_model()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_loss, test_acc = flower_baseline.test(
        model, flower_baseline.load_global_test_data(), device
    )

    print(
        f"[server] round {server_round} — "
        f"accuracy: {round(100 * test_acc, 2)}%  loss: {round(test_loss, 4)}"
    )
    return MetricRecord({"loss": test_loss, "accuracy": test_acc})
