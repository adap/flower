import flwr as fl

from strategy import FedAnalytics

hist = fl.server.start_server(
    server_address="0.0.0.0:8080",
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
