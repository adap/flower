"""timeseries: A Flower / TensorFlow app."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
from datasets import Dataset, load_dataset, load_from_disk
from flwr_datasets import FederatedDataset
from keras import layers
from keras.preprocessing import timeseries_dataset_from_array

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def load_model():
    model = keras.Sequential(
        [
            keras.Input(shape=(12, 1)),
            layers.LSTM(units=12, activation='tanh', return_sequences=True),
            layers.Dropout(rate=0.005),
            layers.LSTM(units=6, activation='tanh'),
            layers.Dense(units=1),
        ]
    )
    model.compile("adam", keras.losses.mean_squared_error, metrics=[keras.losses.mean_squared_error])
    return model

fds = None  # Cache FederatedDataset

def train_test_dataloader(dataset: Dataset, batch_size: int):
    """Divide dataset into train and test sets and return dataloaders."""
    partition_temperature = dataset["T (degC)"][0:32000]
    
    # Create temporal windows for the LSTM neural network
    # from the 12 previous time steps the next step shall be predicted
    input_data = partition_temperature[:-12]
    targets = partition_temperature[12:]
    tf_dataset = timeseries_dataset_from_array(
        input_data, 
        targets, 
        sequence_length=12,
        sequence_stride=1,
        sampling_rate=1,
        shuffle=False,
        batch_size=batch_size)

    # Divide data on each node: 80% train, 20% test of 1000 batches
    tf_test = tf_dataset.take(200)
    tf_train = tf_dataset.skip(200)
    
    return tf_train, tf_test

def load_sim_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load partition jena-climate data."""
    global fds
    if fds is None:
        fds = FederatedDataset(
            dataset="sayanroy058/Jena-Climate",
            partitioners={"train": num_partitions},
            shuffle=False
        )
    partition = fds.load_partition(partition_id, "train")
    partition.set_format("numpy")
    
    tf_train, tf_test = train_test_dataloader(partition, batch_size)
    
    return tf_train, tf_test

    
def load_local_data(data_path: str, batch_size: int):
    """Load local data."""
    # Load dataset from disk
    local_data = load_from_disk(data_path)
    # Divide dataset into train and test sets
    tf_train, tf_test = train_test_dataloader(local_data, batch_size)
    return tf_train, tf_test
    
    
def load_centralized_dataset():
    """Load test set and return dataloader."""
    test_data = load_dataset(path="sayanroy058/Jena-Climate")
    test_data = test_data["train"]["T (degC)"][0:5000]
    # Create temporal windows for the LSTM neural network
    # from the 12 previous time steps the next step shall be predicted
    input_data = test_data[:-12]
    targets = test_data[12:]
    test_dataloader = timeseries_dataset_from_array(
        input_data, 
        targets, 
        sequence_length=12,
        sequence_stride=1,
        sampling_rate=1,
        shuffle=False,
        batch_size=1)
    return test_dataloader
    
    
def test(model, test_dataloader):
    loss, accuracy  = model.evaluate(test_dataloader)
    return loss, accuracy 