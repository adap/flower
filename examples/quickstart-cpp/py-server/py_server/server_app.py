"""Python ServerApp for C++ quickstart."""

import numpy as np
from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from py_server.strategy import FedAvgCpp, weights_to_parameters


def server_fn(context: Context):
    initial_weights = [
        np.array([1.0, 2.0], dtype=np.float64),
        np.array([3.0], dtype=np.float64),
    ]
    initial_parameters = weights_to_parameters(initial_weights)
    strategy = FedAvgCpp(initial_parameters=initial_parameters)

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Define strategy
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
