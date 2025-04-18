import copy
import toml
from omegaconf import OmegaConf

from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from tracefl.dataset import (
    get_clients_server_data,
)
from tracefl.fls import FLSimulation
from tracefl.utils import set_exp_key
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("tracefl_experiment.log", mode="w")
    ]
)

def server_fn(context: Context):

    config = toml.load("./tracefl/resnet.toml") #replace with transformer to run that
    cfg = OmegaConf.create(config)
    cfg.tool.tracefl.exp_key = set_exp_key(cfg)

    ds_dict = get_clients_server_data(cfg)

    sim = FLSimulation(
        copy.deepcopy(cfg),
        context.run_config["fraction-fit"],
        context.run_config["num-server-rounds"],
        context.run_config["local-epochs"],
        context.run_config["min-evaluate"],
    )
    sim.set_server_data(ds_dict["server_testdata"])
    sim.set_clients_data(ds_dict["client2data"])
    sim.set_strategy()

    return ServerAppComponents(
        strategy=sim.strategy,
        config=ServerConfig(num_rounds=context.run_config["num-server-rounds"]),
    )


app = ServerApp(server_fn=server_fn)
