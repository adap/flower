"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""
# these are the basic packages you'll need here
# feel free to remove some if aren't needed

from flwr.server.client_manager import SimpleClientManager
import flwr as fl
import torch
import logging 

from fedsmoo.dataset import load_datasets
from fedsmoo.utils import *
from fedsmoo import client
from fedsmoo.server import *
from fedsmoo.strategy import *
from flwr.server.server import Server

import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf", config_name="fedavg", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # 1. Print parsed config
    print(OmegaConf.to_yaml(cfg))

    #-----------------------------------------------------------------------------------------

    # 2. Prepare your dataset
    # here you should call a function in datasets.py that returns whatever is needed to:
    # (1) ensure the server can access the dataset used to evaluate your model after
    # aggregation
    # (2) tell each client what dataset partitions they should use (e.g. a this could
    # be a location in the file system, a list of dataloader, a list of ids to extract
    # from a dataset, it's up to you)
        
    trainloaders, valloaders, testloader = load_datasets(
        config=cfg.dataset,
        num_clients=cfg.num_clients,
        batch_size=cfg.batch_size,
        val_ratio=0.01)

    #------------------------------------------------------------------------------------------

    # 3. Define your clients
    # Define a function that returns another function that will be used during
    # simulation to instantiate each individual client
    # client_fn = client.<my_function_that_returns_a_function>()

    client_cfg = cfg.client

    if cfg.method == "FedSMOO":
        client_fn = client.gen_client_fn_FedSMOO(
        local_epochs=client_cfg.local_epochs,
        learning_rate=client_cfg.learning_rate,
        weight_decay=client_cfg.weight_decay,
        sch_step=client_cfg.sch_step,
        sch_gamma=client_cfg.sch_gamma,
        alpha=client_cfg.alpha,
        lr_decay=client_cfg.lr_decay,
        sam_lr=cfg.sam_lr,
        trainloaders=trainloaders,
        valloaders=valloaders,
        model=cfg.model,)

    elif cfg.method == "FedAvg":
        client_fn = client.gen_client_fn_FedAvg(
        local_epochs=client_cfg.local_epochs,
        learning_rate=client_cfg.learning_rate,
        lr_decay=client_cfg.lr_decay,
        trainloaders=trainloaders,
        valloaders=valloaders,
        model=cfg.model,
        stragglers=client_cfg.stragglers,
        num_rounds=cfg.num_rounds,
        num_clients=cfg.num_clients,) 

    else:
        print("write a suitable message")
        client_fn = None
    
    #----------------------------------------------------------------------------------------------

    # 4. Define your strategy
    # pass all relevant argument (including the global dataset used after aggregation,
    # if needed by your method.)

    # testloader = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluate_fn = gen_evaluate_fn(testloader, device, cfg.model) 
    
    strategy = instantiate(cfg.strategy,
                           evaluate_fn=evaluate_fn)

    
    #-----------------------------------------------------------------------------------------------
    # 4.5 Create a custom server?

    client_manager = SimpleClientManager()
    
    if cfg.method == "FedSMOO": 
        server = instantiate(cfg.server,
                        net = instantiate(cfg.model).to(device),
                        sam_lr = cfg.sam_lr,
                        client_manager = client_manager,
                        strategy = strategy)
    
    elif cfg.method == "FedAvg":
        server = Server(client_manager = client_manager,
                        strategy = strategy)
    
    else:
        # use strategy to initialize the default server
        print("write a suitable message")
        server = None
        
    
    #-----------------------------------------------------------------------------------------------
    # 5. Start Simulation
    
    print("\n---------------------- starting simulation --------------------------\n")

    history = fl.simulation.start_simulation(
                    client_fn=client_fn,
                    num_clients=cfg.num_clients,
                    server=server,
                    config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
                    client_resources={"num_cpus": cfg.client_resources.num_cpus,
                                    "num_gpus": cfg.client_resources.num_gpus,},
                    strategy=strategy)
    #------------------------------------------------------------------------------------------------

    # 6. Save your results
    # Here you can save the `history` returned by the simulation and include
    # also other buffers, statistics, info needed to be saved in order to later
    # on generate the plots you provide in the README.md. You can for instance
    # access elements that belong to the strategy for example:
    # data = strategy.get_my_custom_data() -- assuming you have such method defined.
    # Hydra will generate for you a directory each time you run the code. You
    # can retrieve the path to that directory with this:
    
    save_path = HydraConfig.get().runtime.output_dir
    accuracy_test = history.metrics_centralized["accuracy"]
    plot_fn(save_path, accuracy_test)


if __name__ == "__main__":
    main()