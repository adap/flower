"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""
import random

# these are the basic packages you'll need here
# feel free to remove some if aren't needed
import flwr as fl
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from moon import client, server
from moon.dataset import get_dataloader
from moon.dataset_preparation import partition_data


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
    # here you should call a function in datasets.py that returns whatever is needed to:
    # (1) ensure the server can access the dataset used to evaluate your model after
    # aggregation
    # (2) tell each client what dataset partitions they should use (e.g. a this could
    # be a location in the file system, a list of dataloader, a list of ids to extract
    # from a dataset, it's up to you)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    (
        _,
        _,
        _,
        _,
        net_dataidx_map,
    ) = partition_data(
        dataset=cfg.dataset.name,
        datadir=cfg.dataset.dir,
        partition=cfg.dataset.partition,
        num_clients=cfg.num_clients,
        beta=cfg.dataset.beta,
    )

    _, test_dl, _, _ = get_dataloader(
        dataset=cfg.dataset.name,
        datadir=cfg.dataset.dir,
        train_bs=cfg.batch_size,
        test_bs=32,
    )

    trainloaders = []
    testloaders = []
    for idx in range(cfg.num_clients):
        train_dl, test_dl, _, _ = get_dataloader(
            cfg.dataset.name, cfg.dataset.dir, cfg.batch_size, 32, net_dataidx_map[idx]
        )

        trainloaders.append(train_dl)
        testloaders.append(test_dl)
    # 3. Define your clients
    # Define a function that returns another function that will be used during
    # simulation to instantiate each individual client
    # client_fn = client.<my_function_that_returns_a_function>()
    client_fn = client.gen_client_fn(
        trainloaders=trainloaders,
        testloaders=testloaders,
        cfg=cfg,
    )

    # get function that will executed by the strategy's evaluate() method
    # Set server's device
    device = cfg.server_device
    server.gen_evaluate_fn(test_dl, device=device, cfg=cfg)

    # # get a function that will be used to construct the config that the client's
    # # fit() method will received
    # def get_on_fit_config():
    #     def fit_config_fn(server_round: int):
    #         # resolve and convert to python dict
    #         fit_config = OmegaConf.to_container(cfg.fit_config, resolve=True)
    #         fit_config["curr_round"] = server_round  # add round info
    #         return fit_config

    #     return fit_config_fn

    # 4. Define your strategy
    # pass all relevant argument (including the global dataset used after aggregation,
    # if needed by your method.)
    # strategy = instantiate(cfg.strategy, <additional arguments if desired>)
    strategy = fl.server.strategy.FedAvg(fraction_fit=cfg.fraction_fit)
    # 5. Start Simulation
    # history = fl.simulation.start_simulation(<arguments for simulation>)
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
    # Here you can save the `history` returned by the simulation and include
    # also other buffers, statistics, info needed to be saved in order to later
    # on generate the plots you provide in the README.md. You can for instance
    # access elements that belong to the strategy for example:
    # data = strategy.get_my_custom_data() -- assuming you have such method defined.
    # Hydra will generate for you a directory each time you run the code. You
    # can retrieve the path to that directory with this:
    # save_path = HydraConfig.get().runtime.output_dir

    # Experiment completed. Now we save the results and
    # generate plots using the `history`
    print("................")
    print(history)

    # Hydra automatically creates an output directory
    # Let's retrieve it and save some results there
    # save_path = HydraConfig.get().runtime.output_dir

    # # save results as a Python pickle using a file_path
    # # the directory created by Hydra for each run
    # save_results_as_pickle(history, file_path=save_path, extra_results={})

    # # plot results and include them in the readme
    # strategy_name = strategy.__class__.__name__
    # file_suffix: str = (
    #     f"_{strategy_name}"
    #     f"{'_iid' if cfg.dataset_config.iid else ''}"
    #     f"{'_balanced' if cfg.dataset_config.balance else ''}"
    #     f"{'_powerlaw' if cfg.dataset_config.power_law else ''}"
    #     f"_C={cfg.num_clients}"
    #     f"_B={cfg.batch_size}"
    #     f"_E={cfg.num_epochs}"
    #     f"_R={cfg.num_rounds}"
    #     f"_mu={cfg.mu}"
    #     f"_strag={cfg.stragglers_fraction}"
    # )
