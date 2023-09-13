"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""
import client
import flwr as fl
import hydra
import models
import numpy as np
import torch
from client_manager_HeteroFL import client_manager_HeteroFL
from dataset import load_datasets
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from strategy import HeteroFL
from utils import Model_rate_manager, get_global_model_rate, preprocess_input


@hydra.main(config_path="conf", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # 1. Print parsed config
    # print(OmegaConf.to_yaml(cfg))

    # 2. Prepare your dataset
    # here you should call a function in datasets.py that returns whatever is needed to:
    # (1) ensure the server can access the dataset used to evaluate your model after
    # aggregation
    # (2) tell each client what dataset partitions they should use (e.g. a this could
    # be a location in the file system, a list of dataloader, a list of ids to extract
    # from a dataset, it's up to you)

    # 3. Define your clients
    # Define a function that returns another function that will be used during
    # simulation to instantiate each individual client
    # client_fn = client.<my_function_that_returns_a_function>()

    # 4. Define your strategy
    # pass all relevant argument (including the global dataset used after aggregation,
    # if needed by your method.)
    # strategy = instantiate(cfg.strategy, <additional arguments if desired>)

    # 5. Start Simulation
    # history = fl.simulation.start_simulation(<arguments for simulation>)

    # 6. Save your results
    # Here you can save the `history` returned by the simulation and include
    # also other buffers, statistics, info needed to be saved in order to later
    # on generate the plots you provide in the README.md. You can for instance
    # access elements that belong to the strategy for example:
    # data = strategy.get_my_custom_data() -- assuming you have such method defined.
    # Hydra will generate for you a directory each time you run the code. You
    # can retrieve the path to that directory with this:
    # save_path = HydraConfig.get().runtime.output_dir

    # print config structured as YAML
    print(OmegaConf.to_yaml(cfg))

    trainloaders, label_split, valloaders, testloader = load_datasets(
        config=cfg.dataset_config, num_clients=cfg.num_clients, seed=cfg.seed
    )

    model_config = preprocess_input(cfg.model, cfg.dataset_config)

    # send this array(client_model_rate_mapping) as an argument to client_manager and client
    model_split_rate = {"a": 1, "b": 0.5, "c": 0.25, "d": 0.125, "e": 0.0625}
    model_split_mode = cfg.control.model_split_rate
    model_mode = cfg.control.model_mode

    client_to_model_rate_mapping = [0 for _ in range(cfg.num_clients)]
    model_rate_manager = Model_rate_manager(
        model_split_mode, model_split_rate, model_mode
    )
    client_manager = client_manager_HeteroFL(
        model_rate_manager, client_to_model_rate_mapping , client_label_split = label_split
    )

    # # for i in range(cfg.num_clients):
    #     # client_to_model_rate_mapping[i]

    # prepare function that will be used to spawn each client
    client_fn = client.gen_client_fn(
        model=model_config["model_name"],
        data_shape=model_config["data_shape"],
        hidden_layers=model_config["hidden_layers"],
        classes_size=model_config["classes_size"],
        norm = model_config["norm"],
        global_model_rate=model_split_rate[get_global_model_rate(model_mode)],
        num_clients=cfg.num_clients,
        client_to_model_rate_mapping=client_to_model_rate_mapping,
        num_epochs=cfg.num_epochs,
        trainloaders=trainloaders,
        label_split=label_split,
        valloaders=valloaders,
    )

    strategy = HeteroFL(
        net=model_config["model_name"](
            model_rate=model_split_rate[get_global_model_rate(model_mode)],  # to be modified
            data_shape=model_config["data_shape"],
            hidden_layers=model_config["hidden_layers"],
            classes_size=model_config["classes_size"],
            norm = model_config["norm"]
            # device =
        ),
        fraction_fit=0.1,
        fraction_evaluate=0.1,
        min_fit_clients=10,
        min_evaluate_clients=10,
        min_available_clients=cfg.num_clients,
    )

    history = fl.simultion.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        # client_resources = client,
        client_manager=client_manager,
        strategy=strategy,
    )

    # plot grpahs using history and save the results.

    # fl.server.strategy.fedavg
    # fl.simulation.start_simulation
    # fl.server.start_server


if __name__ == "__main__":
    main()
