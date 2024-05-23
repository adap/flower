from typing import List, Tuple

import numpy as np
from client import app as client_app
from client import client_fn

import flwr as fl
from flwr.common import Metrics

STATE_VAR = "timestamp"


# Define metric aggregation function
def record_state_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Ensure that timestamps are monotonically increasing."""
    states = []
    for _, m in metrics:
        # split string and covert timestamps to float
        states.append([float(tt) for tt in m[STATE_VAR].split(",")])

    for client_state in states:
        if len(client_state) == 1:
            continue
        deltas = np.diff(client_state)
        assert np.all(
            deltas > 0
        ), f"Timestamps are not monotonically increasing: {client_state}"

    return {STATE_VAR: states}


strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=record_state_metrics
)


# Define ServerAppp
server_app = fl.server.ServerApp(
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)


# Run with FlowerNext
fl.simulation.run_simulation(
    server_app=server_app, client_app=client_app, num_supernodes=2
)
