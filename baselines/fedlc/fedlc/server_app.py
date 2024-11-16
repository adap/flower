"""fedlc: A Flower Baseline."""

import csv
import json
from datetime import datetime
from logging import DEBUG, INFO
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from fedlc.model import initialize_model
from flwr.common import Context
from flwr.common.logger import log
from flwr.common.typing import NDArrays, Scalar, UserConfig
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from .dataset import get_transformed_ds
from .model import set_parameters, test
from .strategy import CheckpointedFedAvg, CheckpointedFedProx

RESULTS_FILE = "eval_results.csv"


# From https://github.com/adap/flower/pull/3908
def create_run_dir(config: UserConfig) -> Path:
    """Create a directory where to save results from this run."""
    strategy = str(config["strategy"])
    use_lc = bool(config["use-logit-correction"])
    run_type = f"{strategy}_{'lc' if use_lc else 'no_lc'}"

    # Create output directory given current timestamp
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    save_path = Path.cwd() / f"results/{run_type}/{run_dir}"
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

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    proximal_mu = context.run_config["proximal-mu"]
    save_params_every = context.run_config["save-params-every"]
    log(DEBUG, f"Saving params every {save_params_every} rounds")
    
    use_lc = bool(context.run_config["use-logit-correction"])
    if use_lc:
        log(INFO, "Using logit correction")
    else:
        log(INFO, "NOT using logit correction")

    num_classes = int(context.run_config["num-classes"])
    num_channels = int(context.run_config["num-channels"])
    model_name = str(context.run_config["model-name"])
    dataset = str(context.run_config["dataset"])
    partition_by = str(context.run_config["dataset-partition-by"])
    batch_size = int(context.run_config["batch-size"])

    net = initialize_model(model_name, num_channels, num_classes)

    test_ds = load_dataset(dataset, split="test")
    testloader = DataLoader(
        get_transformed_ds(test_ds, dataset, partition_by),
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

    clients_per_round = int(context.run_config["clients-per-round"])

    strategy_kwargs: Dict[str, Any] = {
        "fraction_fit": 0.00001,  # we want no. of clients to be determined by min_fit_clients
        "fraction_evaluate": 0,
        "min_evaluate_clients": 0,
        "min_fit_clients": clients_per_round,
        "min_available_clients": clients_per_round,
        "evaluate_fn": evaluate,
        "accept_failures": False,
    }

    strategy = str(context.run_config["strategy"])
    use_last_checkpoint = bool(context.run_config["use-last-checkpoint"])

    # Define strategy
    if strategy == "fedprox":
        strategy = CheckpointedFedProx(
            net=net,
            run_config=context.run_config,
            proximal_mu=float(proximal_mu),
            use_last_checkpoint=use_last_checkpoint,
            **strategy_kwargs,
        )
    else:
        # default to FedAvg
        strategy = CheckpointedFedAvg(
            net=net,
            run_config=context.run_config,
            use_last_checkpoint=use_last_checkpoint,
            **strategy_kwargs,
        )
    config = ServerConfig(num_rounds=int(num_rounds))

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
