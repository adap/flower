"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""

from functools import partial
from logging import DEBUG, INFO

import flwr as fl
import hydra
from flwr.common.logger import log
import torch

from feddebug import  utils
from feddebug.server import get_strategy
from feddebug.client import CNNFlowerClient
from feddebug.dataset import ClientsAndServerDatasets
from feddebug.models import initialize_model

utils.seed_everything(786)

def gen_client_func(cfg, client2data, cid):
    """Give the new client."""
    model_dict = initialize_model(cfg.model.name, cfg.dataset)
    args = {
        "cid": cid,
        "model_dict": model_dict,
        "client_data_train": client2data[cid],
        "device": torch.device(cfg.device),
    }
    client = CNNFlowerClient(args).to_client()
    return client


def run_simulation(cfg):
    """Run the simulation."""
    if cfg.total_malicious_clients:
        cfg.malicious_clients_ids = list(range(cfg.total_malicious_clients))
    

    cfg.malicious_clients_ids = [f"{c}" for c in cfg.malicious_clients_ids]

    cfg.exp_key = utils.set_exp_key(cfg)

    log(INFO, f" ***********  Starting Experiment: {cfg.exp_key} ***************")

    log(DEBUG, f"Simulation Configuration: {cfg}")

    ds_prep = ClientsAndServerDatasets(cfg)
    ds_dict = ds_prep.get_data()

    client_app = fl.client.ClientApp(client_fn=partial(gen_client_func, cfg, ds_dict["client2data"]))
    server_config = fl.server.ServerConfig(num_rounds=cfg.num_rounds)
    server_app = fl.server.ServerApp(
        config=server_config, strategy=get_strategy(cfg, ds_dict["server_testdata"]))
    fl.simulation.run_simulation(server_app=server_app, client_app=client_app,
                                 num_supernodes=cfg.num_clients, backend_config=utils.config_sim_resources(cfg))

    log(INFO, f"Training Complete for Experiment: {cfg.exp_key} ")


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg) -> None:
    """Run the baseline."""
    run_simulation(cfg)


if __name__ == "__main__":
    main()
