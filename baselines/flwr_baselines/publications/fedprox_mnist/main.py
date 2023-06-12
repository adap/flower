"""Runs CNN federated learning for MNIST dataset."""


from pathlib import Path

import flwr as fl
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from hydra.utils import instantiate

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
        iid=cfg.iid,
        balance=cfg.balance,
        learning_rate=cfg.learning_rate,
        stragglers=cfg.stragglers_fraction,
        model=cfg.model,
    )

    evaluate_fn = utils.gen_evaluate_fn(testloader, DEVICE, cfg.model)



    # instantiate strategy according to config. Here we pass other arguments
    # that are only defined at run time.
    strategy = instantiate(cfg.strategy,
                           evaluate_fn=evaluate_fn,
                           # function (in this case anonymous) that will be used to generate the config to be send
                           # to clients to do fit(). Even though FedProx will always send them `proximal_mu`,
                           # this is not the case with other strategies. Therefore we include it.
                           #! Make sure mu is only non-zero for FedProx
                           on_fit_config_fn=lambda curr_round: {"curr_round": curr_round, "proximal_mu": cfg.mu},
                           evaluate_metrics_aggregation_fn=utils.weighted_average)


    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
    )

    file_suffix: str = (
        f"{'_iid' if cfg.iid else ''}"
        f"{'_balanced' if cfg.balance else ''}"
        f"_C={cfg.num_clients}"
        f"_B={cfg.batch_size}"
        f"_E={cfg.num_epochs}"
        f"_R={cfg.num_rounds}"
        f"_mu={cfg.mu}"
        f"_strag={cfg.stragglers_fraction}"
    )

    # ensure save directory exists
    save_path = Path(cfg.save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    print("................")
    print(history)

    np.save(
        save_path / Path(f"hist{file_suffix}"),
        history,  # type: ignore
    )

    utils.plot_metric_from_history(
        history,
        cfg.save_path,
        (file_suffix),
    )


if __name__ == "__main__":
    main()
