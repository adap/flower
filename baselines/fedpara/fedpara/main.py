"""Main script for running FedPara."""

import flwr as fl
import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fedpara import client, server, utils
from fedpara.dataset import load_datasets
from fedpara.utils import get_parameters, seed_everything


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # 1. Print parsed config
    print(OmegaConf.to_yaml(cfg))

    seed_everything(cfg.seed)

    # 2. Prepare dataset
    train_loaders, test_loader = load_datasets(
        config=cfg.dataset_config,
        num_clients=cfg.num_clients,
        batch_size=cfg.batch_size,
        partition_equal=True,
    )

    # 3. Define clients
    client_fn = client.gen_client_fn(
    train_loaders=train_loaders,
    model=cfg.model,
    num_epochs=cfg.num_epochs,
    args={"device": cfg.client_device},
    )

    evaluate_fn = server.gen_evaluate_fn(
        test_loader=test_loader, model=cfg.model, device=cfg.server_device
    )

    def get_on_fit_config():
        def fit_config_fn(server_round: int):
            fit_config = OmegaConf.to_container(cfg.hyperparams, resolve=True)
            fit_config["curr_round"] = server_round
            return fit_config

        return fit_config_fn

    net_glob = instantiate(cfg.model)


    # 4. Define strategy
    strategy = instantiate(
        cfg.strategy,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=get_on_fit_config(),
        initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(net_glob)),
        net_glob=net_glob,
    )

    # 5. Start Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={
            "num_cpus": cfg.client_resources.num_cpus,
            "num_gpus": cfg.client_resources.num_gpus,
        },
        ray_init_args={
            "ignore_reinit_error": True,
            "include_dashboard": False,
            "num_cpus": 12,
            "num_gpus": 1,
            "_memory": 100*1024*1024*1024,
        },
    )


    # 6. Save results
    save_path = HydraConfig.get().runtime.output_dir
    file_suffix = "_".join(
        [
            repr(strategy),
            cfg.dataset_config.name,
            f"{cfg.seed}",
            f"{cfg.dataset_config.alpha}",
            f"{cfg.num_clients}",
            f"{cfg.num_rounds}",
            f"{cfg.clients_per_round}",
        ]
    )

    utils.plot_metric_from_history(
        hist=history, save_plot_path=save_path, suffix=file_suffix, cfg=cfg
    )


if __name__ == "__main__":
    main()
