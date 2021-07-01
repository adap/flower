import os

import flwr as fl
from flwr.server.strategy.fedavg import FedAvg
import tensorflow as tf

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def main() -> None:
    # Define Flower client
    class CifarClient(fl.client.NumPyClient):
        def __init__(self,
            cid: str,
            fed_dir: str            
        ) -> None:
            super().__init__()

            # Load and compile Keras model
            self.model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
            self.model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
            
            # Load CIFAR-10 dataset
            (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()


        def get_parameters(self):  # type: ignore
            return self.model.get_weights()

        def fit(self, parameters, config):  # type: ignore
            self.model.set_weights(parameters)
            self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32, steps_per_epoch=10, verbose=2)
            return self.model.get_weights(), len(self.x_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            self.model.set_weights(parameters)
            loss, accuracy = self.model.evaluate(self.x_test, self.y_test, steps=3, verbose=2)
            return loss, len(self.x_test), {"accuracy": accuracy}

    # Start Flower simulation
    fl.server.start_ray_simulation(
        pool_size=1_000_000,
        data_partitions_dir="",  # path where data partitions for each client exist
        client_resources={'num_cpus': 4},  # compute/memory resources for each client
        client_type=CifarClient,
        strategy=FedAvg(
            fraction_fit=0.00001,
            fraction_eval=0.000005,
        ),
        config={"num_rounds": 3},
    )


if __name__ == "__main__":
    main()
