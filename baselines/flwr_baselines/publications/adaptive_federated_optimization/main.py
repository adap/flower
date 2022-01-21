from os import chdir, getcwd
from pathlib import Path
from re import I
from typing import Dict

import flwr as fl
import hydra
import numpy as np
from cifar.utils import plot_metric_from_history
from flwr.common.typing import Parameters
from hydra.utils import call, get_original_cwd, instantiate, to_absolute_path
from omegaconf import DictConfig
from torchvision.datasets import CIFAR10


@hydra.main(config_path="conf/cifar10", config_name="config")
def main(cfg: DictConfig) -> None:
    # Make sure we are on the right directory.
    # This will not be necessary in hydra 1.3
    log_dir = getcwd()
    chdir(get_original_cwd())

    # Create federated partitions - checkout the config files for details
    path_original_dataset = Path(to_absolute_path(cfg.root_dir))
    fed_dir = call(
        cfg.gen_federated_partitions, path_original_dataset=path_original_dataset
    )

    # Get centralized evaluation function - see config files for details
    eval_fn = call(cfg.get_eval_fn, path_original_dataset=path_original_dataset)

    # Define client resources and ray configs
    client_resources = {"num_cpus": cfg.cpus_per_client}
    ray_config = {"include_dashboard": cfg.ray_config.include_dashboard}

    on_fit_config_fn = call(
        cfg.gen_on_fit_config_fn, client_learning_rate=cfg.strategy.eta_l
    )

    # select strategy
    initial_parameters: Parameters = call(cfg.get_initial_parameters)
    strategy = instantiate(
        cfg.strategy.init,
        fraction_fit=float(cfg.num_clients_per_round) / cfg.num_total_clients,
        min_fit_clients=cfg.num_clients_per_round,
        min_available_clients=cfg.num_total_clients,
        on_fit_config_fn=on_fit_config_fn,
        eval_fn=eval_fn,
        initial_parameters=initial_parameters,
        accept_failures=False,
    )
    strategy.initial_parameters = initial_parameters

    # start simulation
    if cfg.is_simulation:
        client_fn = call(cfg.get_ray_client_fn, fed_dir=fed_dir)
        hist = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=cfg.num_total_clients,
            client_resources=client_resources,
            num_rounds=cfg.num_rounds,
            strategy=strategy,
            ray_init_args=ray_config,
        )
    else:  # or start server
        hist = fl.server.app.start_server(
            server_address=cfg.server_address,
            server=cfg.server,
            config={"num_rounds": cfg.num_rounds},
            strategy=strategy,
        )

    plot_metric_from_history(
        hist=hist,
        dataset_name=cfg.dataset_name,
        metric_str="accuracy",
        strategy_name=cfg.strategy.name,
        expected_maximum=cfg.strategy.expected_accuracy,
        save_path=Path(to_absolute_path(log_dir))
        / f"{cfg.dataset_name}_{cfg.strategy.name}.png",
    )


if __name__ == "__main__":
    main()
