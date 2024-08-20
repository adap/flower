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

from fed_debug import server, utils
from fed_debug.client import gen_client_func
from fed_debug.dataset import ClientsAndServerDatasetsPrep
from fed_debug.differential_testing import (
    eval_na_threshold,
    run_fed_debug_differential_testing,
)


def _flwr_fl_sim(cfg, client2data, server_data, cache):
    client_app = fl.client.ClientApp(
        client_fn=partial(gen_client_func, cfg, client2data)
    )
    server_config = fl.server.ServerConfig(num_rounds=cfg.strategy.num_rounds)
    server_app = fl.server.ServerApp(
        config=server_config, strategy=server.get_strategy(cfg, server_data, cache)
    )

    fl.simulation.run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=cfg.num_clients,
        backend_config=utils.config_sim_resources(cfg),  # type: ignore
    )


def run_simulation(cfg):
    """Run the simulation."""
    if cfg.multirun_faulty_clients != -1:
        cfg.faulty_clients_ids = list(range(cfg.multirun_faulty_clients))

    cfg.faulty_clients_ids = [f"{c}" for c in cfg.faulty_clients_ids]
    train_cache = Index(cfg.storage.dir + cfg.storage.train_cache_name)

    cfg.exp_key = utils.set_exp_key(cfg)

    log(INFO, f" ***********  Starting Experiment: {cfg.exp_key} ***************")

    if cfg.check_cache:
        if cfg.exp_key in train_cache:
            temp_dict = train_cache[cfg.exp_key]
            if "complete" in temp_dict and temp_dict["complete"]:  # type: ignore
                log(INFO, f"Experiment already completed: {cfg.exp_key}")
                return

    log(DEBUG, f"Simulation Configuration: {cfg}")

    ds_prep = ClientsAndServerDatasetsPrep(cfg)
    ds_dict = ds_prep.get_clients_server_data()
    client2class = ds_dict["client2class"]

    _flwr_fl_sim(cfg, ds_dict["client2data"], ds_dict["server_testdata"], train_cache)

    temp_input = None
    if "pixel_values" in ds_dict["server_testdata"]:
        temp_input = ds_dict["server_testdata"][0]["pixel_values"].clone().detach()
    else:
        temp_input = ds_dict["server_testdata"][0][0].clone().detach()

    train_cache[cfg.exp_key] = {
        "client2class": client2class,
        "train_cfg": cfg,
        "complete": True,
        "input_shape": temp_input.shape,
    }
    log(INFO, f"Simulation Complete for: {cfg.exp_key} ")


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg) -> None:
    """Run the baseline."""
    if not cfg.generate_graphs and not cfg.vary_thresholds:
        run_simulation(cfg)
        time.sleep(1)
        run_fed_debug_differential_testing(cfg)

    if cfg.vary_thresholds:
        eval_na_threshold(cfg)

    if cfg.generate_graphs:
        utils.gen_thresholds_exp_graph(
            cache_path=cfg.storage.dir + cfg.storage.results_cache_name,
            threshold_exp_key=cfg.threshold_variation_exp_key,
        )
        utils.generate_table2_csv(
            cfg.storage.dir + cfg.storage.results_cache_name,
            igonre_keys=[cfg.threshold_variation_exp_key],
        )


if __name__ == "__main__":
    main()
