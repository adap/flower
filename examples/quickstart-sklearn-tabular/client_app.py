import argparse
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from task import set_model_params, get_model_parameters, set_initial_params
import flwr as fl
from flwr.client import NumPyClient
from flwr_datasets import FederatedDataset


class FlowerClient(NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test, unique_labels):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.unique_labels = unique_labels

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        set_model_params(self.model, parameters)
        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)
        accuracy = self.model.score(self.X_train, self.y_train)
        return (
            get_model_parameters(self.model),
            len(self.X_train),
            {"train_accuracy": accuracy},
        )

    def evaluate(self, parameters, config):  # type: ignore
        set_model_params(self.model, parameters)
        y_pred = self.model.predict_proba(self.X_test)
        loss = log_loss(y_test, y_pred, labels=self.unique_labels)
        accuracy = self.model.score(self.X_test, y_test)
        return loss, len(self.X_test), {"test_accuracy": accuracy}

if __name__ == "__main__":
    N_CLIENTS = 3

    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--partition-id",
        type=int,
        choices=range(0, N_CLIENTS),
        required=True,
        help="Specifies the artificial data partition",
    )
    args = parser.parse_args()
    partition_id = args.partition_id

    # Load the partition data
    fds = FederatedDataset(dataset="hitorilabs/iris", partitioners={"train": N_CLIENTS})

    dataset = fds.load_partition(partition_id, "train").with_format("pandas")[:]
    X = dataset[["petal_length", "petal_width", "sepal_length", "sepal_width"]]
    y = dataset["species"]
    unique_labels = fds.load_split("train").unique("species")
    # Split the on edge data: 80% train, 20% test
    X_train, X_test = X[: int(0.8 * len(X))], X[int(0.8 * len(X)) :]
    y_train, y_test = y[: int(0.8 * len(y))], y[int(0.8 * len(y)) :]

    # Create LogisticRegression Model
    model = LogisticRegression(
        penalty="l2",
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )

    # Setting initial parameters, akin to model.compile for keras models
    set_initial_params(model, n_features=X_train.shape[1], n_classes=3)


    # Start Flower client
    fl.client.start_client(
        server_address="0.0.0.0:8080", client=FlowerClient().to_client()
    )
