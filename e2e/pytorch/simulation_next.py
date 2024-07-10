from client import app as client_app

import flwr as fl

# Define ServerAppp
server_app = fl.server.ServerApp(
    config=fl.server.ServerConfig(num_rounds=3),
)


# # Run with FlowerNext
# fl.simulation.run_simulation(
#     server_app=server_app, client_app=client_app, num_supernodes=2
# )


raise ValueError("error")
