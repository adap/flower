import flwr as fl

from src.py.flwr.server.strategy import FedAvgGrad

if __name__ == "__main__":
    fl.server.start_server("[::]:8080",config={"num_rounds": 4},strategy= FedAvgGrad() )

