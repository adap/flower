"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""
# these are the basic packages you'll need here
# feel free to remove some if aren't needed
import pickle
import hydra
from pathlib import Path
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


import flwr as fl

from fedbn.dataset import get_data
from fedbn.client import gen_client_fn
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
    save_path = HydraConfig.get().runtime.output_dir

    # For FedBN clients we need to persist the state of the BN
    # layers across rounds. In Simulation clients are statess
    # so everything not communicated to the server (as it is the
    # case as with params in BN layers of FedBN clients) is lost
    # once a client completes its training. An upcoming version of
    # Flower suports stateful clients
    bn_states = Path(save_path)/"bn_states"
    bn_states.mkdir()

    # 2. Prepare your dataset
    # please ensure you followed the README.md and you downloaded the
    # pre-processed dataset suplied by the authors of the FedBN paper
    client_data_loaders = get_data(cfg.dataset)

    # 3. Define your client generation function
    client_fn = gen_client_fn(client_data_loaders, cfg.client, cfg.model, bn_states)

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

    # save results as a Python pickle using a file_path
    # the directory created by Hydra for each run
    data = {"history": history}
    history_path = f"{str(save_path)}/history.pkl"
    with open(history_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # simple plot
    quick_plot(history_path)


if __name__ == "__main__":
    main()