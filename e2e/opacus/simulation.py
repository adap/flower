import flwr as fl

from client import FlowerClient, Net

def client_fn(cid):
    _ = cid
    model = Net()
    return FlowerClient(model)

hist = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=2,
    config=fl.server.ServerConfig(num_rounds=3),
)
assert (hist.losses_distributed[0][1] / hist.losses_distributed[-1][1]) > 0.98
