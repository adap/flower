import flwr as fl
from flwr.server.strategy.fedavg import FedAvg

from flwr.server.strategy.secagg import SecAgg

# Start Flower server for three rounds of federated learning
if __name__ == "__main__":
    fl.server.start_server("localhost:8080", config={
                           "num_rounds": 1}, strategy=SecAgg())

'''
# Start Flower server for three rounds of federated learning
if __name__ == "__main__":
    fl.server.start_server("localhost:8080", config={
                           "num_rounds": 1}, strategy=FedAvg())'''
