from typing import Any, Callable, Dict, List, Optional, Tuple

import flwr as fl
import tensorflow as tf


def main() -> None:
    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_eval=0.2,
        min_fit_clients=3,
        min_eval_clients=2,
        min_available_clients=10,
        eval_fn=get_eval_fn(),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
    )
    # Start Flower server for four rounds of federated learning
    fl.server.start_server("[::]:8080", config={"num_rounds": 4}, strategy=strategy)


def get_eval_fn():
    """Return an evaluation function for server-side evaluation."""
    # Load data and model here to avoid the overhead of doing in in `evaluate` itself
    (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()

    # Use the last 5k examples as a validation set 
    x_val, y_val = x_train[45000:50000], y_train[45000:50000]

    # Load and compile model
    model = tf.keras.applications.MobileNetV2((32, 32, 3), weights=None, classes=10)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # The `evaluate` function will be called after every round
    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        model.set_weights(weights)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_val, y_val)
        return loss, accuracy

    return evaluate


def fit_config(rnd: int):
    """Return a configuration with fixed batch size and (local) epochs."""
    config = {
        "batch_size": 32,
        "local_epochs": 1 if rnd < 2 else 2,
    }
    return config


def evaluate_config(rnd: int):
    config = {
        "val_steps": 5 if rnd < 4 else 10,
    }
    return config


if __name__ == "__main__":
    main()
