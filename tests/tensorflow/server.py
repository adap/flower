import flwr as fl


# Start Flower server
hist = fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
)

assert (hist.losses_distributed[0][1] / hist.losses_distributed[-1][1]) > 0.98
