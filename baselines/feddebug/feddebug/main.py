"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""

import time
from functools import partial
from logging import DEBUG, INFO

import flwr as fl
import hydra
from diskcache import Index
from flwr.common.logger import log

from feddebug import server, utils
from feddebug.client import gen_client_func
from feddebug.dataset import ClientsAndServerDatasets
# from feddebug.differential_testing import (
#     eval_na_threshold,
#     run_fed_debug_differential_testing,
# )


utils.seed_everything(786)


def _flwr_fl_sim(cfg, client2data, server_data):
    client_app = fl.client.ClientApp(
        client_fn=partial(gen_client_func, cfg, client2data)
    )
    server_config = fl.server.ServerConfig(num_rounds=cfg.strategy.num_rounds)
    server_app = fl.server.ServerApp(
        config=server_config, strategy=server.get_strategy(cfg, server_data)
    )

    fl.simulation.run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=cfg.num_clients,
        backend_config=utils.config_sim_resources(cfg),  # type: ignore
    )


def run_simulation(cfg):
    """Run the simulation."""
    if cfg.total_faulty_clients != -1:
        cfg.faulty_clients_ids = list(range(cfg.total_faulty_clients))

    cfg.faulty_clients_ids = [f"{c}" for c in cfg.faulty_clients_ids]
    train_cache = Index(cfg.storage.dir + cfg.storage.train_cache_name)

    cfg.exp_key = utils.set_exp_key(cfg)

    log(INFO, f" ***********  Starting Experiment: {cfg.exp_key} ***************")

    

    log(DEBUG, f"Simulation Configuration: {cfg}")

    ds_prep = ClientsAndServerDatasets(cfg)
    ds_dict = ds_prep.get_data()

    
    _flwr_fl_sim(cfg, ds_dict["client2data"], ds_dict["server_testdata"])
  
    train_cache[cfg.exp_key] = {
        "train_cfg": cfg,
        "complete": True,
    }
    log(INFO, f"Training Complete for Experiment: {cfg.exp_key} ")


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg) -> None:
    """Run the baseline."""
    if (
        not cfg.generate_thresholds_exp_graph
        and not cfg.vary_thresholds
        and not cfg.generate_table_csv
    ):
        run_simulation(cfg)
    

    if cfg.generate_table_csv:
        utils.generate_table2_csv(cfg.storage.dir + cfg.storage.results_cache_name)

    if cfg.generate_thresholds_exp_graph:
        utils.gen_thresholds_exp_graph(
            cache_path=cfg.storage.dir + cfg.storage.results_cache_name,
            threshold_exp_key=cfg.threshold_variation_exp_key,
        )


if __name__ == "__main__":
    main()
