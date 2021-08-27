import flwr as fl
import os
import tensorflow as tf
import time
from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from multiprocessing import Process
from flwr.server.strategy.sec_agg_fedavg import SecAggFedAvg

# Testing
# Define Flower client

model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


class CifarClient(fl.client.NumPyClient):
    def get_parameters(self):  # type: ignore
        return model.get_weights()

    def fit(self, parameters, config):  # type: ignore
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):  # type: ignore
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}


def test_start_server(sample_num=2, min_num=2, vector_dimension=100000, dropout_value=0):
    fl.server.start_server("localhost:8080", config={
                           "num_rounds": 1, "sec_agg": 1},
                           strategy=SecAggFedAvg(fraction_fit=1, min_available_clients=sample_num,
                                                 sec_agg_param_dict={"min_num": min_num,
                                                                     "test": 1,
                                                                     "test_vector_dimension": vector_dimension,
                                                                     "test_dropout_value": dropout_value}))


def test_start_client(server_address: str,
                      client,
                      grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,):
    fl.client.start_numpy_client(server_address, client, grpc_max_message_length)


def test_start_simulation(sample_num=2, min_num=2, vector_dimension=100000, dropout_value=0):
    """Start a FL simulation."""
    # This will hold all the processes which we are going to create
    processes = []

    # Start the server
    server_process = Process(
        target=test_start_server, args=(
            sample_num, min_num, vector_dimension, dropout_value)
    )
    server_process.start()
    processes.append(server_process)

    # Optionally block the script here for a second or two so the server has time to start
    time.sleep(2)

    # Start all the clients
    for i in range(sample_num):
        client_process = Process(target=test_start_client,
                                 args=("localhost:8080", CifarClient()))
        client_process.start()
        processes.append(client_process)

    # Block until all processes are finished
    for p in processes:
        p.join()


if __name__ == '__main__':
    test_start_simulation()
