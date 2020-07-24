import flwr as fl

if __name__ == "__main__":
    fl.app.server.start_server(config={"num_rounds": 3})
