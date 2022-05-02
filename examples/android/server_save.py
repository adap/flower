from typing import Dict, List, Optional, Tuple, cast
import flwr as fl
import tensorflow as tf
import numpy as np
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    Weights,
)
from flwr.server.strategy.aggregate import aggregate
from functools import reduce
from keras.models import load_model


def create_model():
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(32, 32, 3)),
            tf.keras.layers.Conv2D(6, 5, activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(16, 5, activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=120, activation="relu"),
            tf.keras.layers.Dense(units=84, activation="relu"),
            tf.keras.layers.Dense(units=10, activation="softmax"),
        ]
    )

    model.compile(optimizer='sgd',
                  loss=tf.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=[tf.metrics.SparseCategoricalAccuracy()])

    return model


def ndarray_to_bytes(ndarray: np.ndarray) -> bytes:
    """Serialize NumPy array to bytes."""
    return cast(bytes, ndarray.tobytes())


def weights_to_parameters(weights: Weights) -> Parameters:
    """Convert NumPy weights to parameters object."""
    tensors = [ndarray_to_bytes(ndarray) for ndarray in weights]
    return Parameters(tensors=tensors, tensor_type="numpy.nda")

def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    cifar10 = tf.keras.datasets.cifar10
    (trainImages, trainLabels), (testImages, testLabels) = cifar10.load_data()

    # The `evaluate` function will be called after every round
    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:

        weights[0] = weights[0].reshape((5, 5, 3, 6))
        weights[2] = weights[2].reshape((5, 5, 6, 16))
        weights[4] = weights[4].reshape((1600, 120))
        weights[6] = weights[6].reshape((120, 84))
        weights[8] = weights[8].reshape((84, 10))

        model.set_weights(weights)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(testImages/255, testLabels)
        print("global model, accuracy: {:5.2f}%".format(100 * accuracy))
        return loss, {"accuracy": accuracy}

    return evaluate


# Load and compile model for server-side parameter evaluation
model = create_model()
cifar10 = tf.keras.datasets.cifar10
(trainImages, trainLabels), (testImages, testLabels) = cifar10.load_data()

class FedAvgAndroidSaveAndEvaluate(fl.server.strategy.FedAvgAndroid):
    def weights_to_parameters(self, weights: Weights) -> Parameters:
        """Convert NumPy weights to parameters object."""
        tensors = [self.ndarray_to_bytes(ndarray) for ndarray in weights]
        return Parameters(tensors=tensors, tensor_type="numpy.nda")

    def aggregate(self, results: List[Tuple[Weights, int]]) -> Weights:
        """Compute weighted average."""
        # Calculate the total number of examples used during training
        num_examples_total = sum([num_examples for _, num_examples in results])

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            [layer * num_examples for layer in weights] for weights, num_examples in results
        ]

        for weights, num_examples in results:
            mod = create_model()

            weights[0] = weights[0].reshape((5, 5, 3, 6))
            weights[2] = weights[2].reshape((5, 5, 6, 16))
            weights[4] = weights[4].reshape((1600, 120))
            weights[6] = weights[6].reshape((120, 84))
            weights[8] = weights[8].reshape((84, 10))

            mod.set_weights(weights)
            mod.compile("sgd", "sparse_categorical_crossentropy",
                          metrics=["accuracy"])

            loss, acc = mod.evaluate(testImages/255, testLabels)
            print("client model, accuracy: {:5.2f}%".format(100 * acc))

        # Compute average weights of each layer
        weights_prime: Weights = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
        return weights_prime

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
            # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
            # Convert results
        weights_results = [
            (self.parameters_to_weights(fit_res.parameters), fit_res.num_examples)
            for client, fit_res in results
        ]

        print(f"Saving round {rnd} weights...")
        np.savez(f"round-{rnd}-weights.npz",aggregate(weights_results))

        return self.weights_to_parameters(self.aggregate(weights_results)), {}


def main() -> None:
    strategy = FedAvgAndroidSaveAndEvaluate(
        fraction_fit=1.0,
        fraction_eval=1.0,
        min_fit_clients=10,
        min_eval_clients=10,
        min_available_clients=10,
        eval_fn=get_eval_fn(model),
        # eval_fn=None,
        on_fit_config_fn=fit_config,
        initial_parameters=None,
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server("[::]:8999", config={
                           "num_rounds": 100}, strategy=strategy)


def fit_config(rnd: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 5,
    }
    return config


if __name__ == "__main__":
    main()
