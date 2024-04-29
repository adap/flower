import flwr as fl

from client import client_fn
from client import app as client_app

hist = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=2,
    config=fl.server.ServerConfig(num_rounds=3),
)

assert hist.losses_distributed[-1][1] == 0 or (hist.losses_distributed[0][1] / hist.losses_distributed[-1][1]) >= 0.98

# Define ServerAppp
server_app = fl.server.ServerApp(
    config=fl.server.ServerConfig(num_rounds=3),
)


# Run with FlowerNext
fl.simulation.run_simulation(server_app=server_app, client_app=client_app, num_supernodes=2)