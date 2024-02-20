import flwr as fl


# Start Flower server
hist = fl.server.start_driver(
    server_address="0.0.0.0:9091",
    config=fl.server.ServerConfig(num_rounds=3),
)

assert hist.losses_distributed[-1][1] == 0
