import flwr as fl
import numpy as np
from strategy import Strategy
from client import FlowerClient
from task import get_partitions_and_label

partitions, label = get_partitions_and_label()


def client_fn(cid):
    return FlowerClient(cid, partitions[int(cid)]).to_client()


# Start Flower server
hist = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=3,
    config=fl.server.ServerConfig(num_rounds=1000),
    strategy=Strategy(label),
)

np.save("_static/results/hist.npy", hist)
