import sys

sys.path.insert(0, "../../src/py")
import flwr as fl

strategy = fl.server.strategy.FedAvgDp()
# Start Flower server
fl.server.start_server(server_address="[::]:8080", config={"num_rounds": 3}, strategy=strategy)
