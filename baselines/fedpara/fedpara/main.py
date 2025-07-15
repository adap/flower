"""Main script for running FedPara."""

from typing import Dict

import flwr as fl
import hydra
from flwr.common import Scalar
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fedpara import client, server, utils
from fedpara.dataset import load_datasets
from fedpara.server import weighted_average
from fedpara.utils import (
    get_parameters,
    save_results_as_pickle,
    seed_everything,
    set_client_state_save_path,
)


@hydra.main(config_path="conf", config_name="cifar10", version_base=None)
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
    OmegaConf.to_container(cfg, resolve=True)
    if "state_path" in cfg:
        state_path = set_client_state_save_path(cfg.state_path)
    else:
        state_path = None

    # 2. Prepare dataset
    train_loaders, test_loader = load_datasets(
        config=cfg.dataset_config,
        num_clients=cfg.num_clients,
        batch_size=cfg.batch_size,
    )

    # 3. Define clients
    # In this scheme the responsability of choosing the client is on the client manager

    client_fn = client.gen_client_fn(
        train_loaders=train_loaders,
        test_loader=test_loader,
        model=cfg.model,
        num_epochs=cfg.num_epochs,
        args={"algorithm": cfg.algorithm},
        state_path=state_path,
    )

    def get_on_fit_config():
        def fit_config_fn(server_round: int):
            fit_config: Dict[str, Scalar] = OmegaConf.to_container(
                cfg.hyperparams, resolve=True
            )  # type: ignore
            fit_config["curr_round"] = server_round
            return fit_config

        return fit_config_fn

    net_glob = instantiate(cfg.model)

    # 4. Define strategy
    if cfg.strategy.min_evaluate_clients:
        strategy = instantiate(
            cfg.strategy,
            on_fit_config_fn=get_on_fit_config(),
            initial_parameters=fl.common.ndarrays_to_parameters(
                get_parameters(net_glob)
            ),
            evaluate_metrics_aggregation_fn=weighted_average,
        )

    else:
        evaluate_fn = server.gen_evaluate_fn(
            num_clients=cfg.num_clients,
            test_loader=test_loader,
            model=cfg.model,
            device=cfg.server_device,
        )
        strategy = instantiate(
            cfg.strategy,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=get_on_fit_config(),
            initial_parameters=fl.common.ndarrays_to_parameters(
                get_parameters(net_glob)
            ),
        )

    # 5. Start Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources=cfg.client_resources,
    )
    save_path = HydraConfig.get().runtime.output_dir

    save_results_as_pickle(history, file_path=save_path)

    # 6. Save results
    file_suffix = "_".join([(net_glob).__class__.__name__, f"{cfg.exp_id}"])

    utils.plot_metric_from_history(
        hist=history,
        save_plot_path=save_path,
        suffix=file_suffix,
        cfg=cfg,
        model_size=net_glob.model_size[1],
    )


if __name__ == "__main__":
    main()
