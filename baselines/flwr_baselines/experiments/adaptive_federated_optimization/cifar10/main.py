from pathlib import Path
from typing import Dict
import numpy as np

import flwr as fl
import hydra
from hydra.utils import instantiate, to_absolute_path

from omegaconf import DictConfig
from torchvision.datasets import CIFAR10
from flwr.common.typing import Parameters

from client import (
    CifarRayClient,
    get_eval_fn,
    get_model,
    transforms_test,
)
from utils import (
    partition_and_save,
    torch_model_to_parameters,
)

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    fed_dir = (
        Path(to_absolute_path(cfg.root_dir))
        / f"{cfg.dataset_name}"
        / "partitions"
        / f"{cfg.num_total_clients}"
        / f"{cfg.lda_concentration:.2f}"
    )

    # Create Partitions
    trainset = CIFAR10(root=to_absolute_path(cfg.root_dir), train=True, download=True)
    flwr_trainset = (trainset.data, np.array(trainset.targets, dtype=np.int32))
    partition_and_save(
        dataset=flwr_trainset,
        fed_dir=fed_dir,
        dirichlet_dist=None,
        num_partitions=cfg.num_total_clients,
        concentration=cfg.lda_concentration,
    )

    # Download testset for centralized evaluation
    testset = CIFAR10(
        root=to_absolute_path(cfg.root_dir),
        train=False,
        download=True,
        transform=transforms_test,
    )

    # Define client resources and ray configs
    client_resources = {"num_cpus": cfg.cpus_per_client}
    ray_config = {"include_dashboard": cfg.ray_config.include_dashboard}

    def client_fn(cid: str) -> CifarRayClient:
        # create a single client instance
        return CifarRayClient(cid, fed_dir)

    # Helper functions
    def fit_config(rnd: int) -> Dict[str, str]:
        """Return a configuration with specific client learning rate."""
        local_config = {
            "epoch_global": str(rnd),
            "epochs": str(cfg.epochs_per_round),
            "batch_size": str(cfg.batch_size),
            "client_learning_rate": cfg.strategy.eta_l,
        }
        return local_config

    # select strategy
    initial_parameters = torch_model_to_parameters(get_model())
    strategy = instantiate(
        config=cfg.strategy.build,
        fraction_fit=float(cfg.num_clients_per_round) / cfg.num_total_clients,
        min_fit_clients=cfg.num_clients_per_round,
        min_available_clients=cfg.num_total_clients,
        on_fit_config_fn=fit_config,
        eval_fn=get_eval_fn(testset),
        initial_parameters= list(initial_parameters),
    )

    # start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_total_clients,
        client_resources=client_resources,
        num_rounds=cfg.num_rounds,
        strategy=strategy,
        ray_init_args=ray_config,
    )


if __name__ == "__main__":
    main()
