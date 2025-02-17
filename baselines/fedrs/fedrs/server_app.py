"""fedrs: A Flower Baseline."""

import csv
import json
from datetime import datetime
from logging import INFO
from pathlib import Path
from typing import Dict, Optional, Tuple
import wandb
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
from .utils import get_ds_info

RESULTS_FILE = "eval_results.csv"
PROJECT_NAME = "Flower-Baseline-FedRS"

# From https://github.com/adap/flower/pull/3908
def create_run_dir(config: UserConfig) -> Tuple[Path,Path]:
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
    results_csv_path = save_path / RESULTS_FILE
    with open(results_csv_path, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "loss", "acc"])

    # Create checkpoints directory
    checkpoint_path = save_path / "checkpoints"
    checkpoint_path.mkdir(exist_ok=True)

    return results_csv_path, checkpoint_path


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_csv_path, checkpoint_path = create_run_dir(context.run_config)

    num_rounds = int(context.run_config["num-server-rounds"])

    model_name = str(context.run_config["model-name"])
    dataset = str(context.run_config["dataset"])
    num_classes, partition_by = get_ds_info(dataset)
    net = initialize_model(model_name, num_classes)

    alpha = float(context.run_config["alpha"])
    log(INFO, f"Restricted softmax scaling factor is {alpha}")

    dataset = str(context.run_config["dataset"])
    batch_size = int(context.run_config["batch-size"])

    test_ds = load_dataset(dataset, split="test")
    testloader = DataLoader(
        get_transformed_ds(test_ds, dataset, partition_by, split="test"),
        batch_size=batch_size,
    )

    alg = str(context.run_config["alg"])
    proximal_mu = float(context.run_config["proximal-mu"])
    
    use_wandb = context.run_config["use-wandb"]
    if use_wandb:
        num_shards_per_partition = int(context.run_config["num-shards-per-partition"])
        group_name = f"{dataset}-100-{num_shards_per_partition}"
        run_name = "fedavg"
        if alg == "fedprox":
            run_name = f"fedprox_{proximal_mu}"
        if alpha < 1.0:
            # Using FedRS
            run_name = f"fedrs_{alpha}"

        wandb.init(
            project=PROJECT_NAME,
            group=group_name,
            name=run_name,
        )

    checkpoint_every = int(context.run_config["checkpoint-every"])

    # The `evaluate` function will be called by Flower after every round
    def evaluate(
        server_round: int,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        set_parameters(net, parameters)
        loss, accuracy = test(net, testloader, device)
        log(INFO, f"Server-side evaluation loss {loss} / accuracy {accuracy}")
        
        if use_wandb:
            wandb.log({"loss": loss, "accuracy": accuracy}, step=server_round)
        else:
            with open(results_csv_path, "a", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([server_round, loss, accuracy])
        
        if server_round > 0 and server_round % checkpoint_every == 0:
            checkpoint_file_path = checkpoint_path / f"ckpt_round_{server_round}.path"
            torch.save(net.state_dict(), checkpoint_file_path)
            log(
                INFO,
                f"Saved global model checkpoint to {checkpoint_file_path} on round {server_round}",
            )

        return loss, {"accuracy": accuracy}

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

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
