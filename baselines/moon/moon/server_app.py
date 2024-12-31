"""moon: A Flower Baseline."""

import csv
import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from flwr.common import Context
from flwr.common.typing import NDArrays, Scalar, UserConfig
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from moon.dataset_preparation import get_data_transforms, get_transforms_apply_fn
from moon.models import init_net, test

RESULTS_FILE = "results.csv"


def gen_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
    dataset_name: str,
    model_name: str,
    model_output_dim: int,
    run_dir: Path,
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]
]:
    """Generate the function for centralized evaluation."""

    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # pylint: disable=unused-argument
        net = init_net(dataset_name, model_name, model_output_dim)
        params_dict = zip(net.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        net.to(device)
        accuracy, loss = test(net, testloader, device=device)

        # Append results
        with open(f"{run_dir}/{RESULTS_FILE}", "a", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([server_round, loss, accuracy])

        return loss, {"accuracy": accuracy}

    return evaluate


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
    fields = ["round", "loss", "acc"]
    with open(f"{save_path}/{RESULTS_FILE}", "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)

    return save_path


def server_fn(context: Context) -> ServerAppComponents:
    """Construct ServerAppComponents object to create a ServerApp."""
    run_dir = create_run_dir(context.run_config)

    dataset_name = str(context.run_config["dataset-name"])
    partition_by = context.run_config["dataset-partition-by"]
    # This is the exact same dataset as the one donwloaded by the clients via
    # FlowerDatasets.However, we don't use FlowerDatasets for the server since
    # partitioning it is not needed.
    # We make use of the "test" split only
    global_test_set = load_dataset(dataset_name)["test"]
    _, test_transforms = get_data_transforms(dataset_name=dataset_name)

    batch_size = int(context.run_config["batch-size"])
    transforms_fn = get_transforms_apply_fn(test_transforms, partition_by)
    testloader = DataLoader(
        global_test_set.with_transform(transforms_fn),
        batch_size=batch_size,
    )

    evaluate_fn = gen_evaluate_fn(
        testloader,
        device=str(context.run_config["server-device"]),
        dataset_name=dataset_name,
        model_name=str(context.run_config["model-name"]),
        model_output_dim=int(context.run_config["model-output-dim"]),
        run_dir=run_dir,
    )

    strategy = FedAvg(
        # Clients in MOON do not perform federated evaluation
        # (see the client's evaluate())
        fraction_fit=float(context.run_config["fraction-fit"]),
        fraction_evaluate=0.0,
        evaluate_fn=evaluate_fn,
    )

    config = ServerConfig(num_rounds=int(context.run_config["num-server-rounds"]))

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
