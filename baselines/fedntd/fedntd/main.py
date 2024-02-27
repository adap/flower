"""Runs CNN federated learning for MNIST dataset."""

from typing import Dict

import flwr as fl
import hydra
from datasets.utils.logging import disable_progress_bar
from flwr_datasets import FederatedDataset
from omegaconf import DictConfig, OmegaConf

from fedntd.client import get_client_fn
from fedntd.server import get_evaluate_fn


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    print(OmegaConf.to_yaml(cfg))

    NUM_CLIENTS = 2

    mnist_fds = FederatedDataset(dataset="mnist", partitioners={"train": NUM_CLIENTS})
    centralized_testset = mnist_fds.load_full("test")

    def fit_config(server_round: int) -> Dict[str, float]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            "epochs": 1,
            "lr": 0.01,
        }
        return config

    strategy = fl.server.strategy.FedAvg(
        on_fit_config_fn=fit_config,
        evaluate_fn=get_evaluate_fn(centralized_testset),
    )

    disable_progress_bar()

    fl.simulation.start_simulation(
        client_fn=get_client_fn(mnist_fds),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
        actor_kwargs={"on_actor_init_fn": disable_progress_bar},
    )


if __name__ == "__main__":
    main()
