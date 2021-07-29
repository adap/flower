import flwr as fl
from flwr.server.strategy import FedAvg
import tensorflow as tf


def initial_parameters():
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model.get_weights()


def main():
    strategy = FedAvg(initial_parameters=initial_parameters())
    fl.server.start_server(
        "0.0.0.0:8080", strategy=strategy, config={"num_rounds": 3},
    )


# Start Flower server for three rounds of federated learning
if __name__ == "__main__":
    main()
