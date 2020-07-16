import flwr as fl

fl.app.server.start_server(config={"num_rounds": 3})
