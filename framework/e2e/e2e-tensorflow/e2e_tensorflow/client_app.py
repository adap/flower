import os

import tensorflow as tf

from flwr.app import Context
from flwr.client import NumPyClient, start_client
from flwr.clientapp import ClientApp

# Set subset sizes
TRAIN_SUBSET_SIZE = 100
TEST_SUBSET_SIZE = 10

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define one fast preprocessing function
@tf.function
def preprocess_sample(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def preprocess(split, subset_size):
    return (
        split.take(subset_size)
        .map(preprocess_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(32)
        .prefetch(tf.data.AUTOTUNE)
    )


def load_data():
    """Load and preprocess CIFAR-10 data."""
    # Load CIFAR-10 using tf.keras.datasets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    
    # Preprocess datasets
    ds_train = preprocess(ds_train, TRAIN_SUBSET_SIZE)
    ds_test = preprocess(ds_test, TEST_SUBSET_SIZE)
    
    return ds_train, ds_test


# Load model (MobileNetV2, CIFAR-10)
model = tf.keras.applications.MobileNetV2(
    input_shape=(32, 32, 3), classes=10, weights=None
)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])


# Define Flower client
class FlowerClient(NumPyClient):
    def __init__(self, ds_train, ds_test):
        self.ds_train = ds_train
        self.ds_test = ds_test
    
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(self.ds_train, epochs=1, batch_size=32)
        return model.get_weights(), len(self.ds_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(self.ds_test)
        return loss, len(self.ds_test), {"accuracy": accuracy}


def client_fn(context: Context):
    ds_train, ds_test = load_data()
    return FlowerClient(ds_train, ds_test).to_client()


app = ClientApp(
    client_fn=client_fn,
)

if __name__ == "__main__":
    # Start Flower client
    ds_train, ds_test = load_data()
    start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(ds_train, ds_test).to_client()
    )
