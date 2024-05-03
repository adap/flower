import os

import flwr as fl
import tensorflow as tf

SUBSET_SIZE = 1000

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def get_model():
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model

model = get_model()
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, y_train = x_train[:SUBSET_SIZE], y_train[:SUBSET_SIZE]
x_test, y_test = x_test[:SUBSET_SIZE], y_test[:SUBSET_SIZE]


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        return loss, len(x_test), {"accuracy": accuracy}


def client_fn(cid):
    return FlowerClient().to_client()


app = fl.client.ClientApp(
    client_fn=client_fn,
)


if __name__ == "__main__":
    # Start Flower client
    fl.client.start_client(server_address="127.0.0.1:8080", client=FlowerClient().to_client())
