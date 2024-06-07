import os
import argparse
from typing import Dict, List, Tuple

import tensorflow as tf

import flwr as fl
from flwr.common import Metrics
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth

from datasets import Dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner

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
ALPHA = 1000

# Define features to employ
RELEVANT_FEATURES = ['Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare']

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
        client_dataset = dataset.load_partition(int(cid))

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
def preprocess_data(data):
    """Preprocess the dataset."""
    # Select relevant features
    features = data[RELEVANT_FEATURES].copy()
    
    # Convert 'Sex' to binary values
    features['Sex'] = features['Sex'].map({'male': 0, 'female': 1})
    
    # Fill missing 'Age' values with the mean age
    val_miss_age = features['Age'].mean()
    features['Age'].fillna(val_miss_age, inplace=True)
    
    # Scale the features
    scaler = StandardScaler()
    features = pd.DataFrame(scaler.fit_transform(features), columns=RELEVANT_FEATURES)

    data = pd.concat([features,data['Survived']], axis=1)
    
    return data

# Load dataset from Hugging Face
titanic_cds = load_dataset("julien-c/titanic-survival")

# Convert to pandas to preprocess
titanic_cds = titanic_cds['train'].to_pandas()
titanic_cds = preprocess_data(titanic_cds)

# Split the data into training and testing sets
Xy_train, Xy_test = train_test_split(titanic_cds, test_size=0.2, random_state=RANDOM_STATE)

# Set train and test data in Dataset format
train_cds = {"features": Xy_train[RELEVANT_FEATURES].values.tolist(), "labels": Xy_train["Survived"].values.tolist()}
train_cds = pd.DataFrame(train_cds)
train_cds = Dataset.from_pandas(train_cds)

centralized_testset = {"features": Xy_test[RELEVANT_FEATURES].values.tolist(), "labels": Xy_test["Survived"].values.tolist()}
centralized_testset = pd.DataFrame(centralized_testset)
centralized_testset = Dataset.from_pandas(centralized_testset).to_tf_dataset(
    columns="features", label_cols="labels", batch_size=64
)

# Set partitioner to federate data
partitioner = DirichletPartitioner(num_partitions=NUM_CLIENTS, partition_by="labels", alpha=ALPHA)
partitioner.dataset = train_cds

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
    client_fn=get_client_fn(partitioner),
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
    client_fn=get_client_fn(partitioner),
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
