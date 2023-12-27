"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""

import hydra
import torch
import sys
import os
import flwr
from omegaconf import DictConfig, OmegaConf
from client import FedPMClient, DenseClient
import logging

from dataset import get_data_loaders
from strategy import FedPMStrategy, DenseStrategy


get_client = {
    'fedpm': FedPMClient,
    'dense': DenseClient
}


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
    if not cfg.get('simulation').get('verbose'):
        logging.disable()

    # 2. Prepare your dataset
    # here you should call a function in datasets.py that returns whatever is needed to:
    # (1) ensure the server can access the dataset used to evaluate your model after
    # aggregation
    # (2) tell each client what dataset partitions they should use (e.g. this could
    # be a location in the file system, a list of dataloader, a list of ids to extract
    # from a dataset, it's up to you)

    trainloaders, valloaders, testloader = get_data_loaders(
        dataset=cfg.dataset.name,
        nclients=NUM_CLIENTS,
        batch_size=cfg.datatset.minibatch_size,
        classes_pc=cfg.datset.classes_pc,
        split=cfg.dataset.split,
        data_path=cfg.datset.data_path
    )

    # 3. Define your clients
    # Define a function that returns another function that will be used during
    # simulation to instantiate each individual client

    def client_fn(cid) -> flwr.client.Client:
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]

        return get_client[cfg.simulation.client_type](
            params=cfg,
            client_id=cid,
            train_data_loader=trainloader,
            test_data_loader=valloader,
            device=DEVICE,
        )

    # 4. Define your strategy
    # pass all relevant argument (including the global dataset used after aggregation,
    # if needed by your method.)
    strategy = get_strategy[cfg.simulation.client_type](
        params=cfg,
        global_data_loader=testloader,
        fraction_fit=cfg.simulation.fraction_fit,
        fraction_evaluate=cfg.simulation.fraction_evaluate,
        min_fit_clients=0,
        min_evaluate_clients=0,
        min_available_clients=0,
    )

    # 5. Start Simulation
    history = flwr.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=flwr.server.ServerConfig(num_rounds=cfg.simulation.n_rounds),
        strategy=strategy,
        ray_init_args={
            "ignore_reinit_error": cfg.ray_init_args.ignore_reinit_error,
            "include_dashboard": cfg.ray_init_args.ray_init_args,
            "num_cpus": cfg.ray_init_args.num_cpus,
            "num_gpus": cfg.ray_init_args.num_gpus,
            "local_mode": cfg.ray_init_args.local_mode
        }
    )

    # 6. Save your results
    # Here you can save the `history` returned by the simulation and include
    # also other buffers, statistics, info needed to be saved in order to later
    # on generate the plots you provide in the README.md. You can for instance
    # access elements that belong to the strategy for example:
    # data = strategy.get_my_custom_data() -- assuming you have such method defined.
    # Hydra will generate for you a directory each time you run the code. You
    # can retrieve the path to that directory with this:
    # save_path = HydraConfig.get().runtime.output_dir


sys.path.append(os.getcwd())


# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
cfg = OmegaConf.load('conf/base.yaml')
NUM_CLIENTS = cfg.simulation.n_clients

get_strategy = {
    'fedpm': FedPMStrategy,
    'dense': DenseStrategy
}

main()

