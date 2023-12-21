import tensorflow as tf
from flwr_datasets import FederatedDataset
import logging

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)     # Create logger for the module


def load_data(data_sampling_percentage=0.0005,batch_size=32,client_id=1,total_clients=2):
    """
     Load federated dataset partition based on client ID.

    Args:
        batch_size (int): Batch size for training and evaluation.
        client_id (int): Unique ID for the client.
        total_clients (int): Total number of clients.
        data_sampling_percentage (float): Percentage of the dataset to use for training.

    Returns:
        Tuple of TensorFlow datasets for training and evaluation.
    """

    logger.info("Loaded federated dataset partition for client %s", client_id)
    logger.info("total_clients in load_data %s", total_clients)

    # Download and partition dataset
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": total_clients})
    partition = fds.load_partition(client_id-1, "train")
    partition.set_format("numpy")

    # Divide data on each client: 80% train, 20% test
    partition = partition.train_test_split(test_size=0.2)
    x_train, y_train = partition["train"]["img"] / 255.0, partition["train"]["label"]
    x_test, y_test = partition["test"]["img"] / 255.0, partition["test"]["label"]

    # Convert the datasets to tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # Calculate subset size for train dataset
    train_subset_size = int(len(x_train) * data_sampling_percentage)

    # Shuffle and subset data
    train_dataset = train_dataset.shuffle(buffer_size=len(x_train)).take(train_subset_size)
    test_dataset = test_dataset.shuffle(buffer_size=len(x_test))

    # Batch data
    train_dataset = train_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    # Optimize datasets for performance
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    logger.info("Created data generators with batch size: %s", batch_size)
    logger.info("Created data generators with train_subset_size: %s", train_subset_size)
    
    return train_dataset, test_dataset
