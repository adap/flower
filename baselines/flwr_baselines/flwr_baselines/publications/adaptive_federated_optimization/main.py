"""Runs AdaptiveFederated Optimization for CIFAR10/100."""

from os import chdir
from pathlib import Path

import flwr as fl
import hydra
from flwr.common.typing import Parameters
from flwr.server import ServerConfig
from hydra.utils import call, get_original_cwd, instantiate, to_absolute_path
from omegaconf import DictConfig


@hydra.main(config_path="conf/cifar10", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """General-purpose main function that receives cfg from Hydra."""
    # Make sure we are on the right directory.
    # This will not be necessary in hydra 1.3
    chdir(get_original_cwd())

    # Create federated partitions - checkout the config files for details
    path_original_dataset = Path(to_absolute_path(cfg.root_dir))
    fed_dir = call(
        cfg.gen_federated_partitions, path_original_dataset=path_original_dataset
    )

    # Get centralized evaluation function - see config files for details
    evaluate_fn = call(cfg.get_eval_fn, path_original_dataset=path_original_dataset)

    # Define client resources and ray configs
    client_resources = {
        "num_cpus": cfg.cpus_per_client,
        "num_gpus": cfg.gpus_per_client,
    }
    ray_config = {"include_dashboard": cfg.ray_config.include_dashboard}

    on_fit_config_fn = call(
        cfg.gen_on_fit_config_fn, client_learning_rate=cfg.strategy.eta_l
    )

    # select strategy
    initial_parameters: Parameters = call(cfg.get_initial_parameters)
    strategy = instantiate(
        cfg.strategy.init,
        fraction_fit=float(cfg.num_clients_per_round) / cfg.num_total_clients,
        fraction_evaluate=0.0,
        min_fit_clients=cfg.num_clients_per_round,
        min_evaluate_clients=0,
        min_available_clients=cfg.num_total_clients,
        on_fit_config_fn=on_fit_config_fn,
        evaluate_fn=evaluate_fn,
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
            config=ServerConfig(num_rounds=cfg.num_rounds),
            strategy=strategy,
            ray_init_args=ray_config,
        )
    else:  # or start server
        hist = fl.server.app.start_server(
            server_address=cfg.server_address,
            server=cfg.server,
            config=ServerConfig(num_rounds=cfg.num_rounds),
            strategy=strategy,
        )

    # Plot results
    call(cfg.plot_results, hist=hist)


if __name__ == "__main__":
    main()
