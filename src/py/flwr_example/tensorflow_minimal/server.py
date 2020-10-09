import flwr as fl

# Start Flower server for three rounds of federated learning
fl.server.start_server(config={"num_rounds": 3})
