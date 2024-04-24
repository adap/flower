"""Run FL with frozen, pre-trained models."""

import pickle
from pathlib import Path

import flwr as fl
import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fedpft.client import generate_client_fn


# pylint: disable=too-many-locals
@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run federated learning with frozen, pre-trained models.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # Print Config
    print(OmegaConf.to_yaml(cfg))

    # Set device
    device = torch.device(cfg.device)

    # Prepare dataset
    trainloaders, testloaders = instantiate(
        cfg.dataset,
        transform=cfg.model.transform,
        image_input_size=cfg.model.image_input_size,
    ).get_loaders()

    # Define clients
    client_fn = generate_client_fn(
        client_cfg=cfg.client,
        trainloaders=trainloaders,
        testloaders=testloaders,
        feature_extractor=instantiate(cfg.model.feature_extractor),
        num_classes=cfg.dataset.num_classes,
        device=device,
    )

    # Setup strategy
    strategy = instantiate(cfg.strategy)

    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": cfg.num_cpus, "num_gpus": cfg.num_gpus},
    )

    # Save results
    accuracy_per_round = history.metrics_distributed["accuracy"]
    print(accuracy_per_round)
    save_path = HydraConfig.get().runtime.output_dir

    strategy_name = strategy.__class__.__name__

    def format_variable(x):
        return f"{x!r}" if isinstance(x, bytes) else x

    file_suffix: str = (
        f"_{format_variable(strategy_name)}"
        f"_{format_variable(cfg.dataset.name)}"
        f"_clients={format_variable(cfg.num_clients)}"
        f"_rounds={format_variable(cfg.num_rounds)}"
        f"_finalacc={format_variable(accuracy_per_round[-1][1]):.2f}"
    )
    filename = "results" + file_suffix + ".pkl"

    print(f">>> Saving {filename}")
    results_path = Path(save_path) / filename
    results = {"history": history}

    with open(str(results_path), "wb") as hist_file:
        pickle.dump(results, hist_file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
