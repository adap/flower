import flwr as fl

if __name__ == "__main__":
    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3))
