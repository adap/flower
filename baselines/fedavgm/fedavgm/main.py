"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""
import pickle
from pathlib import Path

import flwr as fl
import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from fedavgm.client import generate_client_fn
from fedavgm.dataset import partition, prepare_dataset
from fedavgm.server import get_evaluate_fn


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
    
    # 2. Prepare your dataset
    x_train, y_train, x_test, y_test, input_shape, num_classes = prepare_dataset(
        cfg.dataset.fmnist
    )

    # assert input_shape == cfg.dataset.input_shape, "Conflict on input_shape"
    # assert num_classes == cfg.dataset.num_classes, "Conflict on num_classes"

    partitions = partition(
        x_train, y_train, cfg.num_clients, cfg.dataset.concentration, num_classes
    )

    print(f">>> [Model]: Num. Classes {num_classes} | Input shape: {input_shape}")

    # 3. Define your clients
    client_fn = generate_client_fn(
        partitions,
        cfg.model
    )

    # 4. Define your strategy
    strategy = instantiate(cfg.strategy, 
                           evaluate_fn=get_evaluate_fn(cfg.model, x_test, y_test, cfg.num_rounds),
                           )

    # 5. Start Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
    )

    # 6. Save your results
    # save_path = HydraConfig.get().runtime.output_dir
    # results_path = Path(save_path) / 'results.pkl'
    # results = {"history" : history}

    # with open(str(results_path), 'wb') as h:
    #     pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
