import flwr as fl


# Start Flower server
fl.server.start_server(
    server_address="[::]:8080",
    config={"num_rounds": 3},
)
