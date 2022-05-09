import flwr as fl
import numpy as np
from fedavg_cpp import FedAvgCpp, weights_to_parameters

# Start Flower server for three rounds of federated learning
if __name__ == "__main__":
    model_size = 2
    initial_weights = [
        np.array([1.0, 2.0], dtype=np.float64),
        np.array([3.0], dtype=np.float64),
    ]
    initial_parameters = weights_to_parameters(initial_weights)
    strategy = FedAvgCpp(initial_parameters=initial_parameters)
    fl.server.start_server(
        server_address="127.0.0.1:8888",
        config={"num_rounds": 5},
        strategy=strategy,
    )
