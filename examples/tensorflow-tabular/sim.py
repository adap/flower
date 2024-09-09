import os
import argparse
from typing import Dict, List, Tuple

import tensorflow as tf

import flwr as fl
from flwr.common import Metrics
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth

from datasets import Dataset, concatenate_datasets
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner

import numpy as np
import pandas as pd

from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

parser = argparse.ArgumentParser(description="Flower Simulation with Tensorflow/Keras for tabular dataset")

parser.add_argument(
    "--num_cpus",
    type=int,
    default=1,
    help="Number of CPUs to assign to a virtual client",
)
parser.add_argument(
    "--num_gpus",
    type=float,
    default=0.0,
    help="Ratio of GPU memory to assign to a virtual client",
)

VERBOSE = 0
NUM_CLIENTS = 5
RANDOM_STATE = 42
NUM_ROUNDS = 30

# Define Alpha to use for DirichletPartitioner
ALPHA = 0.5

# Define features to employ
RELEVANT_FEATURES = ['Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare']

# Define labels (target) name to employ
LABEL_NAME = 'Survived'

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainset, valset, num_features) -> None:
        # Create model
        self.model = get_model(num_features)
        self.trainset = trainset
        self.valset = valset

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.trainset, epochs=1, verbose=VERBOSE)
        return self.model.get_weights(), len(self.trainset), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.valset, verbose=VERBOSE)
        return loss, len(self.valset), {"accuracy": acc}


def get_model(num_features):
    """Constructs a simple model architecture suitable for Titanic dataset."""
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(32, input_dim=num_features, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ]
    )
    model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    return model


def get_client_fn(dataset: FederatedDataset):
    """Return a function to construct a client.

    The VirtualClientEngine will execute this function whenever a client is sampled by
    the strategy to participate.
    """

    def client_fn(cid: str) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""

        # Extract partition for client with id = cid
        client_dataset = dataset[int(cid)]

        # Now let's split it into train (90%) and validation (10%)
        client_dataset_splits = client_dataset.train_test_split(test_size=0.1, seed=RANDOM_STATE)

        trainset = client_dataset_splits["train"].to_tf_dataset(
            columns="features", label_cols="labels", batch_size=64
        )

        valset = client_dataset_splits["test"].to_tf_dataset(
            columns="features", label_cols="labels", batch_size=64
        )

        # Extract the number of features
        element_spec = trainset.element_spec
        num_features = element_spec[0].shape[1]
        
        # Create and return client
        return FlowerClient(trainset, valset, num_features).to_client()

    return client_fn


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics, i.e. those returned by
    the client's evaluate() method."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def get_evaluate_fn(testset: Dataset):
    """Return an evaluation function for server-side (i.e. centralised) evaluation."""

    # The `evaluate` function will be called after every round by the strategy
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ):
        # Extract the number of features
        element_spec = testset.element_spec
        num_features = element_spec[0].shape[1]
        model = get_model(num_features)  # Construct the model
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(testset, verbose=VERBOSE)
        return loss, {"accuracy": accuracy}

    return evaluate

# Preprocess the data
def preprocess_data(dataset):
    """Preprocess the dataset."""
    # Select relevant features and target
    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in RELEVANT_FEATURES + [LABEL_NAME]])
    # print(dataset)

    # Convert 'Sex' to binary values
    dataset = dataset.map(lambda x: {'Sex': 0 if x['Sex'] == 'male' else 1})

    # Fill missing 'Age' values with the mean age
    mean_age = np.mean([x['Age'] for x in dataset if x['Age'] is not None])
    dataset = dataset.map(lambda x: {'Age': x['Age'] if x['Age'] is not None else mean_age})

    # Extract features and target
    features = np.array([tuple(x[feature] for feature in RELEVANT_FEATURES) for x in dataset])
    
    target = [x[LABEL_NAME] for x in dataset]

    # Scale the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Convert scaled features back to Dataset and add target column
    dataset = Dataset.from_dict({
        **{RELEVANT_FEATURES[i]: features_scaled[:, i] for i in range(len(RELEVANT_FEATURES))},
        LABEL_NAME: target
    })

    return dataset

# Set partitioner to federate data
partitioner = DirichletPartitioner(num_partitions=NUM_CLIENTS, partition_by=LABEL_NAME,
                                    alpha=ALPHA)

# Load dataset from Hugging Face 
fds = FederatedDataset(dataset="julien-c/titanic-survival", partitioners={"train": partitioner})

# Create a list to store the preprocessed federated dataset
fds_preprocessed = []

# Initialize the centralized dataset
centralized_testset = None

for cid in range(NUM_CLIENTS):
    # Extract partition for client with id = cid
    client_dataset = fds.load_partition(int(cid))
    
    # Preprocess the client dataset
    client_dataset = preprocess_data(client_dataset)

    # Split into train (90%) and validation (10%)
    client_dataset_splits = client_dataset.train_test_split(test_size=0.1, seed=RANDOM_STATE)
    
    # Extract the train data and set it as Dataset format
    train_dataset = client_dataset_splits["train"]
    train_dataset_features = train_dataset.remove_columns([col for col in train_dataset.column_names if col not in RELEVANT_FEATURES])
    train_dataset_labels = train_dataset.remove_columns([col for col in train_dataset.column_names if col not in [LABEL_NAME]])
    train_dataset = {"features": [list(row.values()) for row in train_dataset_features], "labels": [list(row.values()) for row in train_dataset_labels]}
    train_dataset = Dataset.from_dict(train_dataset)

    # Extract the test data and set it as Dataset format
    test_dataset = client_dataset_splits["test"]
    test_dataset_features = test_dataset.remove_columns([col for col in test_dataset.column_names if col not in RELEVANT_FEATURES])
    test_dataset_labels = test_dataset.remove_columns([col for col in test_dataset.column_names if col not in [LABEL_NAME]])
    test_dataset = {"features": [list(row.values()) for row in test_dataset_features], "labels": [list(row.values()) for row in test_dataset_labels]}
    test_dataset = Dataset.from_dict(test_dataset)

    # Append the preprocessed FederatedDataset
    fds_preprocessed.append(train_dataset)

    # Concatenate the test partitions of each client into the centralized_testset
    if centralized_testset is None:
        centralized_testset = test_dataset
    else:
        centralized_testset = concatenate_datasets([centralized_testset, test_dataset])

# Convert centralized dataset to tf tensor format
centralized_testset = centralized_testset.to_tf_dataset(
    columns="features", label_cols="labels", batch_size=64
)

# Create FedAvg strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    min_fit_clients=2,  # Never sample less than 2 clients for training
    min_evaluate_clients=2,  # Never sample less than 2 clients for evaluation
    min_available_clients=int(
        NUM_CLIENTS * 0.75
    ),  # Wait until at least 75 clients are available
    evaluate_metrics_aggregation_fn=weighted_average,  # aggregates federated metrics
    evaluate_fn=get_evaluate_fn(centralized_testset),  # global evaluation function
)


# ClientApp for Flower-Next
client = fl.client.ClientApp(
    client_fn=get_client_fn(fds_preprocessed),
)

# ServerApp for Flower-Next
server = fl.server.ServerApp(
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
)


def main() -> None:
    # Parse input arguments
    args = parser.parse_args()

    # With a dictionary, you tell Flower's VirtualClientEngine that each
    # client needs exclusive access to these many resources in order to run
    client_resources = {
        "num_cpus": args.num_cpus,
        "num_gpus": args.num_gpus,
    }

    # Start simulation
    fl.simulation.start_simulation(
    client_fn=get_client_fn(fds_preprocessed),
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
    client_resources=client_resources,
    actor_kwargs={
        "on_actor_init_fn": enable_tf_gpu_growth  # Enable GPU growth upon actor init.
    },
    )


if __name__ == "__main__":
    # Enable GPU growth in your main process
    enable_tf_gpu_growth()
    main()
