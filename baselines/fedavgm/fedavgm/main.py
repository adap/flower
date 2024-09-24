"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""

import pickle
from pathlib import Path

import flwr as fl
import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fedavgm.client import generate_client_fn
from fedavgm.dataset import partition
from fedavgm.server import get_evaluate_fn


# pylint: disable=too-many-locals
@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    np.random.seed(2020)

    # 1. Print parsed config
    print(OmegaConf.to_yaml(cfg))

    # 2. Prepare your dataset
    x_train, y_train, x_test, y_test, input_shape, num_classes = instantiate(
        cfg.dataset
    )

    partitions = partition(x_train, y_train, cfg.num_clients, cfg.noniid.concentration)

    print(f">>> [Model]: Num. Classes {num_classes} | Input shape: {input_shape}")

    # 3. Define your clients
    client_fn = generate_client_fn(partitions, cfg.model, num_classes)

    # 4. Define your strategy
    evaluate_fn = get_evaluate_fn(
        instantiate(cfg.model), x_test, y_test, cfg.num_rounds, num_classes
    )

    strategy = instantiate(cfg.strategy, evaluate_fn=evaluate_fn)

    # 5. Start Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": cfg.num_cpus, "num_gpus": cfg.num_gpus},
    )

    _, final_acc = history.metrics_centralized["accuracy"][-1]

    # 6. Save your results
    save_path = HydraConfig.get().runtime.output_dir

    strategy_name = strategy.__class__.__name__
    dataset_type = "cifar10" if cfg.dataset.input_shape == [32, 32, 3] else "fmnist"

    def format_variable(x):
        return f"{x!r}" if isinstance(x, bytes) else x

    file_suffix: str = (
        f"_{format_variable(strategy_name)}"
        f"_{format_variable(dataset_type)}"
        f"_clients={format_variable(cfg.num_clients)}"
        f"_rounds={format_variable(cfg.num_rounds)}"
        f"_C={format_variable(cfg.server.reporting_fraction)}"
        f"_E={format_variable(cfg.client.local_epochs)}"
        f"_alpha={format_variable(cfg.noniid.concentration)}"
        f"_server-momentum={format_variable(cfg.server.momentum)}"
        f"_client-lr={format_variable(cfg.client.lr)}"
        f"_acc={format_variable(final_acc):.4f}"
    )

    filename = "results" + file_suffix + ".pkl"

    print(f">>> Saving {filename}...")
    results_path = Path(save_path) / filename
    results = {"history": history}

    with open(str(results_path), "wb") as hist_file:
        pickle.dump(results, hist_file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
