import flwr as fl

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=flwr.server.ServerConfig(num_rounds=3),
)
