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
from fedavgm.datasets import partition
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
    x_train, y_train, x_test, y_test, input_shape, num_classes = instantiate(cfg.dataset)

    partitions = partition(
        x_train, y_train, cfg.num_clients, cfg.noniid.concentration, num_classes
    )

    print(f">>> [Model]: Num. Classes {num_classes} | Input shape: {input_shape}")

    # 3. Define your clients
    client_fn = generate_client_fn(
        partitions,
        cfg.model,
        num_classes
    )
    
    # 4. Define your strategy
    evaluate_fn = get_evaluate_fn(
        instantiate(cfg.model), 
        x_test, 
        y_test, 
        cfg.num_rounds, 
        num_classes
    )

    strategy = instantiate(cfg.strategy, evaluate_fn=evaluate_fn)

    # 5. Start Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
    )

    # 6. Save your results
    save_path = HydraConfig.get().runtime.output_dir
    
    strategy_name = strategy.__class__.__name__
    file_suffix: str = (
        f"_{strategy_name}"
        f"{'_cifar10' if cfg.dataset.input_shape == [32, 32, 3] else '_fmnist'}"
        f"_C={cfg.server.reporting_fraction}"
        f"_E={cfg.client.local_epochs}"
        f"_alpha={cfg.noniid.concentration}"
    )
    
    filename = 'results' + file_suffix + '.pkl'
    results_path = Path(save_path) / filename
    results = {"history" : history}

    with open(str(results_path), 'wb') as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
