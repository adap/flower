"""TraceFL Server Application Module.

This module implements the server-side functionality for the TraceFL federated learning
system. It provides the server implementation that coordinates the federated learning
process and manages the global model.
"""

import copy
import logging
import os
from pathlib import Path

import toml
from omegaconf import OmegaConf

from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from tracefl.dataset import get_clients_server_data
from tracefl.fls import FLSimulation
from tracefl.utils import set_exp_key

# Get the directory where this file is located
current_dir = Path(__file__).parent.parent.parent
log_file_path = current_dir / "tracefl_experiment.log"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(log_file_path), mode="w"),
    ],
    force=True,  # Override any existing logging configuration
)

# Test that logging is working
logging.info("TraceFL Server Application starting...")
logging.info("Log file location: %s", log_file_path)


def server_fn(context: Context):
    """Create and configure a TraceFL server instance.

    The FLSimulation class is used to:
    1. Manage the federated learning simulation process
    2. Handle client and server data setup
    3. Configure and manage the federated learning strategy
    4. Perform provenance analysis and tracking

    Args:
        context: Flower context containing configuration and state

    Returns
    -------
        ServerAppComponents: Configured server components including strategy and config
    """
    # ========== Experiment Configuration ==========
    config_key = os.environ.get("EXPERIMENT", "exp_1")
    print(f"config_key: {config_key}")

    config_path = str(context.run_config[config_key])
    config = toml.load(config_path)
    cfg = OmegaConf.create(config)

    # Override dirichlet_alpha if specified (for exp_3 data distribution experiments)
    dirichlet_alpha = os.environ.get("DIRICHLET_ALPHA")
    if dirichlet_alpha and config_key == "exp_3":
        dirichlet_alpha_float = float(dirichlet_alpha)
        cfg.tool.tracefl.dirichlet_alpha = dirichlet_alpha_float
        cfg.tool.tracefl.data_dist.dirichlet_alpha = dirichlet_alpha_float
        print(f"Overriding dirichlet_alpha to: {dirichlet_alpha_float}")

    # ========== Dataset Preparation ==========
    cfg.tool.tracefl.exp_key = set_exp_key(cfg)
    ds_dict = get_clients_server_data(cfg)

    # ========== FL Simulation Setup ==========
    sim = FLSimulation(
        copy.deepcopy(cfg),
        float(context.run_config["fraction-fit"]),
        int(context.run_config["num-server-rounds"]),
        int(context.run_config["local-epochs"]),
    )

    # Set up server and client data
    sim.set_server_data(ds_dict["server_testdata"])
    sim.set_clients_data(ds_dict["client2data"])

    # Configure the federated learning strategy
    sim.set_strategy()

    # ========== Return Configured Components ==========
    return ServerAppComponents(
        strategy=sim.strategy,
        config=ServerConfig(num_rounds=int(context.run_config["num-server-rounds"])),
    )


app = ServerApp(server_fn=server_fn)
