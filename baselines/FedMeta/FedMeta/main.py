"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""
# these are the basic packages you'll need here
# feel free to remove some if aren't needed
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from strategy import weighted_average
from dataset import load_datasets
from Fedmeta_client_manager import Fedmeta_client_manager
from flwr.common.logger import log
from logging import WARNING
import server

import flwr as fl
import client


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # print config structured as YAML
    print(OmegaConf.to_yaml(cfg))

    # partition dataset and get dataloaders
    dataset, client_list = load_datasets(config=cfg.dataset)
    train_clients, val_clients, test_clients = client_list

    # Check config Clients value
    if cfg.num_clients > len(train_clients):
        raise ImportError(f"Total Clients num is {len(train_clients)}")

    if cfg.min_evaluate_clients > len(val_clients):
        min_evaluate_clients = min(len(val_clients), cfg.min_evaluate_clients)
        log(WARNING, "min_evaluate_clients iis smaller than Validation Clients")

    # prepare function that will be used to spawn each client
    client_fn = client.gen_client_fn(
        num_epochs=cfg.num_epochs,
        dataset=dataset,
        client_list=client_list,
        learning_rate=cfg.learning_rate,
        model=cfg.model,
    )

    # device = cfg.server_device
    # evaluate_fn = server.gen_evaluate_fn(
    #     dataset=dataset,
    #     device=device,
    #     model=cfg.model
    # )


    strategy = instantiate(
        cfg.strategy,
        # evaluate_fn=evaluate_fn,
        evaluate_metrics_aggregation_fn=weighted_average,
        min_evaluate_clients=int(min_evaluate_clients),
    )

    # 5. Start Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        client_manager=Fedmeta_client_manager(),
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


if __name__ == "__main__":
    main()
