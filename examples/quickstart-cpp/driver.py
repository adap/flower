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
    initial_parameters = None
    strategy = FedAvgCpp(initial_parameters=initial_parameters)
    fl.driver.start_driver(
        server_address="0.0.0.0:9091",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
