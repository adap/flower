import flwr as fl
import mnist

from client import FlowerClient

def client_fn(cid):
    _ = cid
    model = mnist.LitAutoEncoder()
    train_loader, val_loader, test_loader = mnist.load_data()

    # Flower client
    return FlowerClient(model, train_loader, val_loader, test_loader)

hist = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=2,
    config=fl.server.ServerConfig(num_rounds=3),
)
assert (hist.losses_distributed[0][1] / hist.losses_distributed[-1][1]) > 0.98
