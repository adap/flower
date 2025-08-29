import os

import tensorflow as tf
import tensorflow_datasets as tfds

from flwr.client import ClientApp, NumPyClient, start_client
from flwr.common import Context

# Set subset sizes
TRAIN_SUBSET_SIZE = 100
TEST_SUBSET_SIZE = 10

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Load CIFAR-10 as supervised (returns (img, label))
ds_train, ds_test = tfds.load("cifar10", split=["train", "test"], as_supervised=True)


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


# Preprocess datasets
ds_train = preprocess(ds_train, TRAIN_SUBSET_SIZE)
ds_test = preprocess(ds_test, TEST_SUBSET_SIZE)


# Load model (MobileNetV2, CIFAR-10)
model = tf.keras.applications.MobileNetV2(
    input_shape=(32, 32, 3), classes=10, weights=None
)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])


# Define Flower client
class FlowerClient(NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(ds_train, epochs=1, batch_size=32)
        return model.get_weights(), len(ds_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(ds_test)
        return loss, len(ds_test), {"accuracy": accuracy}


def client_fn(context: Context):
    return FlowerClient().to_client()


app = ClientApp(
    client_fn=client_fn,
)

if __name__ == "__main__":
    # Start Flower client
    start_client(server_address="127.0.0.1:8080", client=FlowerClient().to_client())
