"""fedrs: A Flower Baseline."""

import csv
import json
from datetime import datetime
from logging import INFO
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from flwr.common import Context
from flwr.common.logger import log
from flwr.common.typing import NDArrays, Scalar, UserConfig
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, FedProx

from .dataset import get_transformed_ds
from .model import initialize_model, set_parameters, test

RESULTS_FILE = "eval_results.csv"


# From https://github.com/adap/flower/pull/3908
def create_run_dir(config: UserConfig) -> Path:
    """Create a directory where to save results from this run."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    save_path = Path.cwd() / f"results/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=False)

    # Save run config as json
    with open(f"{save_path}/run_config.json", "w", encoding="utf-8") as fp:
        json.dump(config, fp)

    # Prepare results.csv
    with open(f"{save_path}/{RESULTS_FILE}", "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "loss", "acc"])

    return save_path


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = create_run_dir(context.run_config)

    num_rounds = context.run_config["num-server-rounds"]

    num_classes = int(context.run_config["num-classes"])
    model_name = str(context.run_config["model-name"])
    net = initialize_model(model_name, num_classes)

    alpha = float(context.run_config["alpha"])
    log(INFO, f"Restricted softmax scaling factor is {alpha}")

    dataset = str(context.run_config["dataset"])
    partition_by = str(context.run_config["dataset-partition-by"])
    batch_size = int(context.run_config["batch-size"])

    test_ds = load_dataset(dataset, split="test")
    testloader = DataLoader(
        get_transformed_ds(test_ds, dataset, partition_by, split="test"),
        batch_size=batch_size,
    )

    # The `evaluate` function will be called by Flower after every round
    def evaluate(
        server_round: int,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        set_parameters(net, parameters)
        loss, accuracy = test(net, testloader, device)
        log(INFO, f"Server-side evaluation loss {loss} / accuracy {accuracy}")

        with open(f"{run_dir}/{RESULTS_FILE}", "a", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([server_round, loss, accuracy])

        return loss, {"accuracy": accuracy}

    alg = str(context.run_config["alg"])
    fraction_fit = float(context.run_config["fraction-fit"])

    # Define strategy
    if alg == "fedavg":
        log(INFO, "Using FedAvg")
        strategy = FedAvg(
            fraction_fit=fraction_fit,  # e.g. 100 clients, select 10 at each round
            fraction_evaluate=0,  # no federated evaluation,
            min_evaluate_clients=0,
            evaluate_fn=evaluate,
            accept_failures=False,
        )
    elif alg == "fedprox":
        proximal_mu = float(context.run_config["proximal-mu"])
        log(INFO, f"Using FedProx with proximal_mu={proximal_mu}")
        strategy = FedProx(
            fraction_fit=fraction_fit,  # e.g. 100 clients, select 10 at each round
            fraction_evaluate=0,  # no federated evaluation,
            min_evaluate_clients=0,
            evaluate_fn=evaluate,
            accept_failures=False,
            proximal_mu=proximal_mu,
        )
    else:
        raise ValueError("Only FedAvg currently supported!")

    config = ServerConfig(num_rounds=int(num_rounds))

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
