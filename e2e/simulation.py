import sys
import flwr as fl

sys.path.append(".")
from client import client_fn

hist = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=2,
    config=fl.server.ServerConfig(num_rounds=3),
)

assert hist.losses_distributed[-1][1] == 0 or (hist.losses_distributed[0][1] / hist.losses_distributed[-1][1]) >= 0.98
