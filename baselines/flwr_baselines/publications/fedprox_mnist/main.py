"""Runs CNN federated learning for MNIST dataset."""

import os
from pathlib import Path

import flwr as fl
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig

from flwr_baselines.publications.fedprox_mnist import client, utils

DEVICE: str = torch.device("cpu")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function to run CNN federated learning on MNIST.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """

    client_fn, testloader = client.gen_client_fn(
        num_epochs=cfg.num_epochs,
        batch_size=cfg.batch_size,
        device=DEVICE,
        num_clients=cfg.num_clients,
        num_rounds=cfg.num_rounds,
        dataset_config=cfg.dataset_config,
        learning_rate=cfg.learning_rate,
        stragglers=cfg.stragglers_fraction,
        model=cfg.model,
    )

    evaluate_fn = utils.gen_evaluate_fn(testloader, DEVICE, cfg.model)


    def get_on_fit_config():
        def fit_config_fn(server_round: int):
            # resolve and convert to python dict
            fit_config = OmegaConf.to_container(cfg.fit_config, resolve=True)
            fit_config['curr_round'] = server_round # add round info
            return fit_config
        return fit_config_fn

    # instantiate strategy according to config. Here we pass other arguments
    # that are only defined at run time.
    strategy = instantiate(cfg.strategy,
                           evaluate_fn=evaluate_fn,
                           on_fit_config_fn=get_on_fit_config(),
                           evaluate_metrics_aggregation_fn=utils.weighted_average)


    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
    )

    strategy_name = strategy.__class__.__name__

    file_suffix: str = (
        f"_{strategy_name}"
        f"{'_iid' if cfg.dataset_config.iid else ''}"
        f"{'_balanced' if cfg.dataset_config.balance else ''}"
        f"{'_powerlaw' if cfg.dataset_config.power_law else ''}"
        f"_C={cfg.num_clients}"
        f"_B={cfg.batch_size}"
        f"_E={cfg.num_epochs}"
        f"_R={cfg.num_rounds}"
        f"_mu={cfg.mu}"
        f"_strag={cfg.stragglers_fraction}"
    )

    # Hydra automatically creates an output directory
    # Let's retrieve it and save some results there
    save_path = HydraConfig.get().runtime.output_dir

    print("................")
    print(history)

    np.save(
        save_path / Path(f"hist{file_suffix}"),
        history,  # type: ignore
    )

    utils.plot_metric_from_history(
        history,
        save_path,
        (file_suffix),
    )


if __name__ == "__main__":
    main()
