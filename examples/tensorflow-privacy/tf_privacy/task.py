import tensorflow as tf

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

fds = None  # Cache FederatedDataset


class Net(tf.keras.Model):
    def __init__(self):
        super(Net, self).__init__()
        # Define layers in the constructor
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu")
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu")
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(128, activation="relu")
        self.fc2 = tf.keras.layers.Dense(10, activation="softmax")

    def call(self, inputs):
        # Define the forward pass
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)


def load_data(partition_id: int, num_partitions: int, batch_size):

    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="ylecun/mnist",
            partitioners={"train": partitioner},
        )

    # fds = FederatedDataset(dataset="mnist", partitioners={"train": 2})
    partition = fds.load_partition(partition_id)
    partition.set_format("numpy")

    # Divide data on each node: 80% train, 20% test
    partition = partition.train_test_split(test_size=0.2, seed=42)
    x_train, y_train = partition["train"]["image"] / 255.0, partition["train"]["label"]
    x_test, y_test = partition["test"]["image"] / 255.0, partition["test"]["label"]

    # Adjust the size of the training dataset to make it evenly divisible by the batch size
    remainder = len(x_train) % batch_size
    if remainder != 0:
        x_train = x_train[:-remainder]
        y_train = y_train[:-remainder]

    return (x_train, y_train), (x_test, y_test)
