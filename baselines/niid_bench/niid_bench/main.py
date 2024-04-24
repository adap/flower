"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""

import os
import pickle

import flwr as fl
import hydra
from flwr.server.client_manager import SimpleClientManager
from flwr.server.server import Server
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf

from niid_bench.dataset import load_datasets
from niid_bench.server_fednova import FedNovaServer
from niid_bench.server_scaffold import ScaffoldServer, gen_evaluate_fn
from niid_bench.strategy import FedNovaStrategy, ScaffoldStrategy


@hydra.main(config_path="conf", config_name="fedavg_base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # 1. Print parsed config
    if "mnist" in cfg.dataset_name:
        cfg.model.input_dim = 256
        # pylint: disable=protected-access
        cfg.model._target_ = "niid_bench.models.CNNMnist"
    print(OmegaConf.to_yaml(cfg))

    # 2. Prepare your dataset
    trainloaders, valloaders, testloader = load_datasets(
        config=cfg.dataset,
        num_clients=cfg.num_clients,
        val_ratio=cfg.dataset.val_split,
    )

    # 3. Define your clients
    client_fn = None
    # pylint: disable=protected-access
    if cfg.client_fn._target_ == "niid_bench.client_scaffold.gen_client_fn":
        save_path = HydraConfig.get().runtime.output_dir
        client_cv_dir = os.path.join(save_path, "client_cvs")
        print("Local cvs for scaffold clients are saved to: ", client_cv_dir)
        client_fn = call(
            cfg.client_fn,
            trainloaders,
            valloaders,
            model=cfg.model,
            client_cv_dir=client_cv_dir,
        )
    else:
        client_fn = call(
            cfg.client_fn,
            trainloaders,
            valloaders,
            model=cfg.model,
        )

    device = cfg.server_device
    evaluate_fn = gen_evaluate_fn(testloader, device=device, model=cfg.model)

    # 4. Define your strategy
    strategy = instantiate(
        cfg.strategy,
        evaluate_fn=evaluate_fn,
    )

    # 5. Define your server
    server = Server(strategy=strategy, client_manager=SimpleClientManager())
    if isinstance(strategy, FedNovaStrategy):
        server = FedNovaServer(strategy=strategy, client_manager=SimpleClientManager())
    elif isinstance(strategy, ScaffoldStrategy):
        server = ScaffoldServer(
            strategy=strategy, model=cfg.model, client_manager=SimpleClientManager()
        )

    # 6. Start Simulation
    history = fl.simulation.start_simulation(
        server=server,
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        client_resources={
            "num_cpus": cfg.client_resources.num_cpus,
            "num_gpus": cfg.client_resources.num_gpus,
        },
        strategy=strategy,
    )

    print(history)

    save_path = HydraConfig.get().runtime.output_dir
    print(save_path)

    # 7. Save your results
    with open(os.path.join(save_path, "history.pkl"), "wb") as f_ptr:
        pickle.dump(history, f_ptr)


if __name__ == "__main__":
    main()
