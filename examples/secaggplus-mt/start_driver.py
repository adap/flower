from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics

from .workflows import SecAggPlusWorkflow

# Define strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Select all available clients
    fraction_evaluate=0.0,  # Disable evaluation
    min_available_clients=2,
)


# Start Flower server
fl.driver.start_driver(
    server_address="0.0.0.0:9091",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
    fl_workflow_factory=SecAggPlusWorkflow(),
)
