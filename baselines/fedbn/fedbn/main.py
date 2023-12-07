"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""

import pickle
from pathlib import Path

import flwr as fl
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fedbn.client import gen_client_fn
from fedbn.dataset import get_data
from fedbn.utils import quick_plot


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

    # Hydra automatically creates an output directory
    # Let's retrieve it and save some results there
    save_path = Path(HydraConfig.get().runtime.output_dir)

    # 2. Prepare your dataset
    # please ensure you followed the README.md and you downloaded the
    # pre-processed dataset suplied by the authors of the FedBN paper
    client_data_loaders = get_data(cfg.dataset)

    # 3. Define your client generation function
    client_fn = gen_client_fn(client_data_loaders, cfg.client, cfg.model, save_path)

    # 4. Define your strategy
    strategy = instantiate(cfg.strategy)

    # 5. Start Simulation
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

    # 6. Save your results
    print("................")
    print(history)

    # Save results as a Python pickle using a file_path
    # the directory created by Hydra for each run
    data = {"history": history}
    history_path = f"{str(save_path)}/history.pkl"
    with open(history_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Simple plot
    quick_plot(history_path)


if __name__ == "__main__":
    main()
