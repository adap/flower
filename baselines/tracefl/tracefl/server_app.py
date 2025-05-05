"""TraceFL Server Application Module.

This module implements the server-side functionality for the TraceFL federated learning
system. It provides the server implementation that coordinates the federated learning
process and manages the global model.
"""

import copy
import logging

import toml
from omegaconf import OmegaConf

from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from tracefl.dataset import (
    get_clients_server_data,
)
from tracefl.fls import FLSimulation
from tracefl.utils import set_exp_key

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("tracefl_experiment.log", mode="w"),
    ],
)


def server_fn(context: Context):
    """Create and configure a TraceFL server instance.

    Args:
        context: Flower context containing configuration and state

    Returns
    -------
        ServerAppComponents: Configured server components including strategy and config
    """
    possible_configs = ["exp_1", "exp_2"]
    config_key = next((k for k in possible_configs if k in context.run_config), "exp_1")
    config_path = str(context.run_config[config_key])
    config = toml.load(config_path)

    cfg = OmegaConf.create(config)
    cfg.tool.tracefl.exp_key = set_exp_key(cfg)

    ds_dict = get_clients_server_data(cfg)

    sim = FLSimulation(
        copy.deepcopy(cfg),
        float(context.run_config["fraction-fit"]),
        int(context.run_config["num-server-rounds"]),
        int(context.run_config["local-epochs"]),
        int(context.run_config["min-evaluate"]),
    )
    sim.set_server_data(ds_dict["server_testdata"])
    sim.set_clients_data(ds_dict["client2data"])
    sim.set_strategy()

    return ServerAppComponents(
        strategy=sim.strategy,
        config=ServerConfig(num_rounds=int(context.run_config["num-server-rounds"])),
    )


app = ServerApp(server_fn=server_fn)
