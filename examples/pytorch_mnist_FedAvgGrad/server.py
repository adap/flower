import flwr as fl

from flwr.server.strategy.FedAvgGrad import FedAvgGrad

# Start Flower server for three rounds of federated learning
if __name__ == "__main__":
    fl.server.start_server("[::]:8080", config={"num_rounds": 5},strategy=FedAvgGrad(),force_final_distributed_eval = True)
