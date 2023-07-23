import os
import configparser
import numpy as np
import math

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import flwr as fl
import tensorflow as tf

from logging import ERROR, INFO, DEBUG
from flwr.common.logger import log

NUM_CLIENTS = 10

class FlwrClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train) -> None:
        super().__init__()
        self.model = model
        split_idx = math.floor(len(x_train) * 0.9)  # Use 10% of x_train for validation
        self.x_train, self.y_train = x_train[:split_idx], y_train[:split_idx]
        self.x_val, self.y_val = x_train[split_idx:], y_train[split_idx:]

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        epochs = config["local_epochs"]

        if epochs is None:
            epochs = 2

        batch_size = config["batch_size"]

        if batch_size is None:
            batch_size = 32
            
        num_samples = config["num_samples"]

        x_train_selected = self.x_train
        y_train_selected = self.y_train

        # Randomly sample num_samples from the training set
        if num_samples is not None:
            idx = np.random.choice(len(self.x_train), num_samples, replace=False)
            x_train_selected = self.x_train[idx]
            y_train_selected = self.y_train[idx]

        print(f"Client training on {len(x_train_selected)} samples, {epochs} epochs, batch size {batch_size}")

        self.model.set_weights(parameters)
        self.model.fit(x_train_selected, y_train_selected, batch_size=batch_size, epochs=epochs, verbose=2)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x_val, self.y_val, verbose=2)
        return loss, len(self.x_val), {"accuracy": acc}


def client_fn(cid: str) -> fl.client.Client:
    # Load model
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Load data partition (divide MNIST into NUM_CLIENTS distinct partitions)
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    partition_size = math.floor(len(x_train) / NUM_CLIENTS)
    idx_from, idx_to = int(cid) * partition_size, (int(cid) + 1) * partition_size
    x_train_cid = x_train[idx_from:idx_to] / 255.0
    y_train_cid = y_train[idx_from:idx_to]

    # Create and return client
    return FlwrClient(model, x_train_cid, y_train_cid)

def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Take batch size, local epochs and number of samples of each client from the server config
    """
    config_file_path = os.path.join(os.path.dirname(__file__), 'config.conf')

    config = {
        "batch_size": 32,
        "local_epochs": 1 if server_round < 2 else 2,
        "num_samples": None,
    }

    if os.path.exists(config_file_path):
        parser = configparser.ConfigParser()
        parser.read(config_file_path)

        if 'TrainingConfig' in parser:
            config["batch_size"] = int(parser['TrainingConfig']['batch_size'])
            config["local_epochs"] = int(parser['TrainingConfig']['local_epochs'])
            config["num_samples"] = int(parser['TrainingConfig']['num_samples'])

    log(
        INFO,
        f"Round {server_round} training config: batch_size={config['batch_size']}, local_epochs={config['local_epochs']}, num_samples={config['num_samples']}"
    )

    print(f"Round {server_round} training config: batch_size={config['batch_size']}, local_epochs={config['local_epochs']}, num_samples={config['num_samples']}")

    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}

def main() -> None:

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.8,
        fraction_evaluate=0.8,
        min_fit_clients=8,
        min_evaluate_clients=8,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn = fit_config,
    )

    # Initialize ray_init_args
    ray_init_args = {
        "ignore_reinit_error": True,
        "include_dashboard": True,
    }

    # Start Flower simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        client_resources={"num_cpus": 1},
        config=fl.server.ServerConfig(num_rounds=5),
        strategy= strategy,
        ray_init_args=ray_init_args,
    )


if __name__ == "__main__":
    main()
