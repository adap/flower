import os

import tensorflow as tf
from datasets import load_dataset

from flwr.client import ClientApp, NumPyClient, start_client
from flwr.common import Context

SUBSET_SIZE = 1000

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load CIFAR-10 from Hugging Face
dataset = load_dataset("uoft-cs/cifar10")
dataset.set_format(type="tensorflow", columns=["img", "label"])


# Define train and test conversions
def convert_to_tf_dataset(split):
    return tf.data.Dataset.from_generator(
        lambda: dataset[split],
        output_signature={
            "img": tf.TensorSpec(shape=(32, 32, 3), dtype=tf.uint8),
            "label": tf.TensorSpec(shape=(), dtype=tf.int64),
        },
    ).map(lambda x: (tf.cast(x["img"], tf.float32) / 255.0, x["label"]))


# Apply and batch/prefetch
ds_train = convert_to_tf_dataset("train").batch(32).prefetch(tf.data.AUTOTUNE)
ds_test = convert_to_tf_dataset("test").batch(32).prefetch(tf.data.AUTOTUNE)

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
