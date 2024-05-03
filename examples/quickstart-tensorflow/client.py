import argparse
import os

from flwr.client import ClientApp, NumPyClient
import tensorflow as tf
from flwr_datasets import FederatedDataset

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Parse arguments
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--partition-id",
    type=int,
    choices=[0, 1, 2],
    default=0,
    help="Partition of the dataset (0, 1 or 2). "
    "The dataset is divided into 3 partitions created artificially.",
)
args, _ = parser.parse_known_args()

# Load model and data (MobileNetV2, CIFAR-10)
model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

# Download and partition dataset
fds = FederatedDataset(dataset="cifar10", partitioners={"train": 3})
partition = fds.load_partition(args.partition_id, "train")
partition.set_format("numpy")

# Divide data on each node: 80% train, 20% test
partition = partition.train_test_split(test_size=0.2, seed=42)
x_train, y_train = partition["train"]["img"] / 255.0, partition["train"]["label"]
x_test, y_test = partition["test"]["img"] / 255.0, partition["test"]["label"]


# Define Flower client
class FlowerClient(NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}


def client_fn(cid: str):
    """Create and return an instance of Flower `Client`."""
    return FlowerClient().to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)


# Legacy mode
if __name__ == "__main__":
    from flwr.client import start_client

    start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient().to_client(),
    )
