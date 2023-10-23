"""Main script for comparison experiments."""

import flwr as fl
import hydra
from omegaconf import DictConfig

from pFedHN.comparison_experiments.client import generate_client_fn
from pFedHN.comparison_experiments.strategy import LogResultsStrategy
from pFedHN.dataset import gen_random_loaders
from pFedHN.utils import set_seed


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    set_seed(42)

    # partition dataset and get dataloaders
    # pylint: disable=unbalanced-tuple-unpacking
    trainloaders, valloaders, testloaders = gen_random_loaders(
        cfg.dataset.data,
        cfg.dataset.root,
        cfg.client.num_nodes,
        cfg.client.batch_size,
        cfg.client.num_classes_per_node,
    )

    # prepare function that will be used to spawn each client
    client_fn = generate_client_fn(trainloaders, testloaders, valloaders, cfg)

    # instantiate strategy according to config
    strategy = LogResultsStrategy(
        fraction_fit=0.1,
        min_fit_clients=5,
        fraction_evaluate=0.0,
        min_evaluate_clients=0,
        min_available_clients=5,
    )

    # Start simulation

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.client.num_nodes,
        config=fl.server.ServerConfig(num_rounds=cfg.client.num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
