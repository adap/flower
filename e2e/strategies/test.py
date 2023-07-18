import flwr as fl
from flwr.server.strategy import FedMedian, FedTrimmedAvg, QFedAvg

from client import FlowerClient


STRATEGY_LIST = [FedMedian, FedTrimmedAvg, QFedAvg]


def client_fn(cid):
    _ = cid
    return FlowerClient()


for Strategy in STRATEGY_LIST:
    hist = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=2,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=Strategy(),
    )
    assert (hist.losses_distributed[0][1] / hist.losses_distributed[-1][1]) > 0.98
