import flwr as fl
import tensorflow as tf
from flwr.common import ndarrays_to_parameters
from flwr.server.strategy import FedMedian, FedTrimmedAvg, QFedAvg, FedAvgM, FedAdam, FedAdagrad, FedYogi

from client import FlowerClient


STRATEGY_LIST = [FedMedian, FedTrimmedAvg, QFedAvg, FedAvgM, FedAdam, FedAdagrad, FedYogi]

init_model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
init_model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

def client_fn(cid):
    _ = cid
    return FlowerClient()

def evaluate(server_round, parameters, config):
    model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    _, (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_test, y_test = x_test[:10], y_test[:10]

    model.set_weights(parameters)

    loss, accuracy = model.evaluate(x_test, y_test)

    # return statistics
    return loss, {"accuracy": accuracy}

for Strategy in STRATEGY_LIST:
    print("Current strategy:", str(Strategy))
    hist = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=2,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=Strategy(evaluate_fn=evaluate, initial_parameters=ndarrays_to_parameters(init_model.get_weights())),
    )
    assert (hist.losses_distributed[0][1] / hist.losses_distributed[-1][1]) > 0.98
