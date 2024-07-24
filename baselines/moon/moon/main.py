"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""

import os
import random
import shutil
from pathlib import Path

# these are the basic packages you'll need here
# feel free to remove some if aren't needed
import flwr as fl
from baselines.moon.moon import client_app
import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from baselines.moon.moon import server_app
from moon.dataset import get_dataloader
from moon.dataset_preparation import partition_data
from moon.utils import plot_metric_from_history


# TODO: delete this filw

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # Clean the model directory to save models for MOON
    if cfg.alg == "moon":
        if os.path.exists(cfg.model.dir):
            shutil.rmtree(cfg.model.dir)
    # 1. Print parsed config
    print(OmegaConf.to_yaml(cfg))

    # 2. Prepare your dataset
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    (
        _,
        _,
        _,
        _,
        net_dataidx_map,
    ) = partition_data(
        dataset=cfg.dataset.name,
        datadir=cfg.dataset.dir,
        partition=cfg.dataset.partition,
        num_clients=cfg.num_clients,
        beta=cfg.dataset.beta,
    )

    _, test_global_dl, _, _ = get_dataloader(
        dataset=cfg.dataset.name,
        datadir=cfg.dataset.dir,
        train_bs=cfg.batch_size,
        test_bs=32,
    )

    trainloaders = []
    testloaders = []
    for idx in range(cfg.num_clients):
        train_dl, test_dl, _, _ = get_dataloader(
            cfg.dataset.name, cfg.dataset.dir, cfg.batch_size, 32, net_dataidx_map[idx]
        )

        trainloaders.append(train_dl)
        testloaders.append(test_dl)
    # 3. Define your clients
    # Define a function that returns another function that will be used during
    # simulation to instantiate each individual client
    client_fn = client_app.gen_client_fn(
        trainloaders=trainloaders,
        testloaders=testloaders,
        cfg=cfg,
    )

    # get function that will executed by the strategy's evaluate() method
    # Set server's device
    device = (
        torch.device("cuda:0")
        if torch.cuda.is_available() and cfg.server_device == "cuda"
        else "cpu"
    )
    evaluate_fn = server_app.gen_evaluate_fn(test_global_dl, device=device, cfg=cfg)

    # 4. Define your strategy
    strategy = fl.server.strategy.FedAvg(
        # Clients in MOON do not perform federated evaluation
        # (see the client's evaluate())
        fraction_fit=cfg.fraction_fit,
        fraction_evaluate=0.0,
        evaluate_fn=evaluate_fn,
    )
    # 5. Start Simulation
    # history = fl.simulation.start_simulation(<arguments for simulation>)
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        client_resources={
            "num_cpus": cfg.client_resources.num_cpus,
            "num_gpus": cfg.client_resources.num_gpus,
        },
        strategy=strategy,
    )
    # remove saved models
    if cfg.alg == "moon":
        shutil.rmtree(cfg.model.dir)

    # 6. Save your results
    # Experiment completed. Now we save the results and
    # generate plots using the `history`
    print("................")
    print(history)

    # Hydra automatically creates an output directory
    # Let's retrieve it and save some results there
    save_path = HydraConfig.get().runtime.output_dir

    # plot results and include them in the readme
    strategy_name = strategy.__class__.__name__
    file_suffix: str = (
        f"_{strategy_name}"
        f"{'_dataset' if cfg.dataset.name else ''}"
        f"_C={cfg.num_clients}"
        f"_B={cfg.batch_size}"
        f"_E={cfg.num_epochs}"
        f"_R={cfg.num_rounds}"
        f"_mu={cfg.mu}"
    )

    plot_metric_from_history(
        history,
        Path(save_path),
        (file_suffix),
    )


if __name__ == "__main__":
    main()
