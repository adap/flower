import os
import flwr as fl
from flwr.server.strategy import FedAvg
import tensorflow as tf


def initial_parameters():
    """Return initial checkpoint parameters"""
    package_directory = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(package_directory, "mlcube/workspace/initial_checkpoint")
    model = tf.keras.models.load_model(filepath)
    return model.get_weights()


def main():
    strategy = FedAvg(initial_parameters=initial_parameters())
    fl.server.start_server(
        "0.0.0.0:8080", strategy=strategy, config={"num_rounds": 3},
    )


# Start Flower server for three rounds of federated learning
if __name__ == "__main__":
    main()
