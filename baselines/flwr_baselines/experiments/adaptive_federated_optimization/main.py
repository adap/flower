from os import chdir, getcwd
from pathlib import Path
from typing import Dict
import numpy as np

import flwr as fl
import hydra
from hydra.utils import call, get_original_cwd, instantiate, to_absolute_path

from omegaconf import DictConfig
from torchvision.datasets import CIFAR10

from cifar.utils import (
    plot_metric_from_history,
)


@hydra.main(config_path="conf/cifar10", config_name="config")
def main(cfg: DictConfig) -> None:
    # Make sure we are on the right directory.
    # This will not be necessary in hydra 1.3
    log_dir = getcwd()
    chdir(get_original_cwd())

    path_original_datasets = Path(to_absolute_path(cfg.root_dir))
    fed_dir = (
        path_original_datasets
        / f"{cfg.dataset}"
        / "partitions"
        / f"{cfg.num_total_clients}"
        / f"{cfg.lda_concentration:.2f}"
    )

    # Create federated partitions - checkout the config files for details
    call(cfg.gen_federated_partitions, path_original_datasets, fed_dir)

    # Get centralized evaluation function - checkout the config files for details
    eval_fn = call(cfg.get_eval_fn, path_original_datasets)

    # Define client resources and ray configs
    client_resources = {"num_cpus": cfg.cpus_per_client}
    ray_config = {"include_dashboard": cfg.ray_config.include_dashboard}

    # helper functions
    # def fit_config(rnd: int) -> Dict[str, str]:
    #    """Return a configuration with specific client learning rate."""
    #    local_config = {
    #        "epoch_global": str(rnd),
    #        "epochs": str(cfg.epochs_per_round),
    #        "batch_size": str(cfg.batch_size),
    #        "client_learning_rate": str(cfg.strategy.eta_l),
    #    }
    #    return local_config

    fit_config_fn = call(cfg.gen_fit_config_fn)

    # select strategy
    initial_parameters = call(cfg.get_initial_parameters)
    strategy = instantiate(
        cfg.strategy.init,
        fraction_fit=float(cfg.num_clients_per_round) / cfg.num_total_clients,
        min_fit_clients=cfg.num_clients_per_round,
        min_available_clients=cfg.num_total_clients,
        on_fit_config_fn=fit_config_fn,
        eval_fn=eval_fn,
        initial_parameters=initial_parameters,
    )
    # start simulation
    client_fn = call(cfg.get_ray_client_fn, fed_dir)
    hist = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_total_clients,
        client_resources=client_resources,
        num_rounds=cfg.num_rounds,
        strategy=strategy,
        ray_init_args=ray_config,
    )

    plot_metric_from_history(
        hist=hist,
        metric_str="accuracy",
        strategy_name=cfg.strategy.name,
        expected_maximum=cfg.strategy.expected_accuracy,
        save_path=Path(to_absolute_path(log_dir)) / f"{cfg.strategy.name}.png",
    )


if __name__ == "__main__":
    main()
