"""Runs CNN federated learning for MNIST dataset."""
import os
from pathlib import Path

import flwr as fl
import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import client
import server
import utils
from dataset import load_datasets
from utils import save_results_as_pickle


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function to run CNN federated learning on MNIST.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """

    # remove possible previous client states
    for filename in os.listdir():
        if filename.endswith("_state.bin") or filename.endswith("_mask.bin"):
            os.remove(filename)

    # print config structured as YAML
    print(OmegaConf.to_yaml(cfg))

    # partition dataset and get dataloaders
    trainloaders, testloader = load_datasets(num_clients=cfg.server.num_clients, iid=cfg.dataset.iid)

    # prepare function that will be used to spawn each client
    client_fn = client.gen_client_fn(
        trainloaders=trainloaders,
        learning_rate=cfg.client.learning_rate,
        model=cfg.model,
        client_device=cfg.client.client_device,
    )

    # using only central evaluation
    evaluate_fn = server.gen_evaluate_fn(
        testloader, device=cfg.server.server_device, model=cfg.model
    )

    # Hydra automatically creates an output directory
    # Let's retrieve it and save some results there
    save_path = HydraConfig.get().runtime.output_dir

    histories = []
    for i in range(cfg.meta.n_repeats):
        # number of epochs per round is determined by probability p
        epochs_per_round = np.random.geometric(p=cfg.server.p, size=cfg.server.num_rounds)

        strategy = instantiate(
            cfg.strategy,
            clients_per_round=cfg.server.clients_per_round,
            epochs_per_round=epochs_per_round,
            eta=cfg.client.eta,
            s=cfg.server.s,
            evaluate_fn=evaluate_fn,
        )

        # Start simulation
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=cfg.server.num_clients,
            config=fl.server.ServerConfig(num_rounds=cfg.server.num_rounds),
            client_resources={
                "num_cpus": cfg.client.client_resources.num_cpus,
                "num_gpus": cfg.client.client_resources.num_gpus,
            },
            strategy=strategy,
        )

        # save results as a Python pickle using a file_path
        # the directory created by Hydra
        save_results_as_pickle(history, file_path=save_path, extra_results={}, default_filename=f"results_{i}.pkl")

        histories.append(history)

    # plot results
    strategy_name = cfg.strategy.__name__
    file_suffix: str = (
        f"_{strategy_name}"
        f"_N={cfg.server.num_clients}"
        f"_C={cfg.server.clients_per_round}"
        f"_K={int(1 / cfg.server.p)}"
        f"_S={cfg.server.s}"
        f"_E={cfg.client.eta}"
        f"_R={cfg.server.num_rounds}"
    )

    utils.plot_metric_from_histories(
        histories,
        Path(save_path),
        file_suffix,
    )


if __name__ == "__main__":
    main()
