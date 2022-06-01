import sys

sys.path.insert(0, "../../src/py")
import flwr as fl


def start_server():
    num_rounds = 3
    strategy = fl.server.strategy.FedAvgDp()
    fl.server.start_server(
        server_address="[::]:8080", config={"num_rounds": num_rounds}, strategy=strategy
    )
