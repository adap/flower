import flwr as fl

from client import client_fn
from strategy import FedAnalytics

hist = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=2,
    config=fl.server.ServerConfig(num_rounds=1),
    strategy=FedAnalytics(),
)
assert hist.metrics_centralized["Aggregated histograms"][1][1] == [
    "Length:",
    "18",
    "46",
    "28",
    "54",
    "32",
    "52",
    "36",
    "12",
    "10",
    "12",
    "Width:",
    "8",
    "14",
    "44",
    "48",
    "74",
    "62",
    "20",
    "22",
    "4",
    "4",
]
