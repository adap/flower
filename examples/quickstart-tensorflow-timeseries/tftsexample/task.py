"""timeseries: A Flower / TensorFlow app."""

import os

import keras
from flwr_datasets import FederatedDataset
from keras import layers
from keras.preprocessing import timeseries_dataset_from_array

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def load_model():
    model = keras.Sequential(
        [
            keras.Input(shape=(12, 1)),
            layers.LSTM(units=12, activation='relu', return_sequences=True),
            layers.Dropout(rate=0.005),
            layers.LSTM(units=6, activation='relu'),
            layers.Dense(units=1, activation="relu"),
        ]
    )
    model.compile("adam", keras.losses.mean_squared_error, metrics=[keras.losses.mean_squared_error])
    return model


fds = None  # Cache FederatedDataset


def load_data(partition_id, num_partitions, batch_size):
    global fds
    if fds is None:
        fds = FederatedDataset(
            dataset="sayanroy058/Jena-Climate",
            partitioners={"train": num_partitions},
            shuffle=False
        )
    partition = fds.load_partition(partition_id, "train")
    partition.set_format("numpy")
    
    partition_temperature = partition["T (degC)"][0:32000]
    
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