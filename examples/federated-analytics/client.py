import argparse
from typing import Dict, List, Tuple

import numpy as np

import flwr as fl

from flwr_datasets import FederatedDataset


column_names = ["sepal_length", "sepal_width"]


def compute_hist(column: List[np.ndarray]) -> np.ndarray:
    freqs, _ = np.histogram(column)
    return freqs


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, X: List[np.ndarray]):
        self.X = X

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        hist_list = []
        # Execute query locally
        for column in range(len(column_names)):
            hist = compute_hist(X[column])
            hist_list.append(hist)
        return (
            hist_list,
            len(self.X[0]),  # get the length of one column
            {},
        )


if __name__ == "__main__":
    N_CLIENTS = 2

    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--node-id",
        type=int,
        choices=range(0, N_CLIENTS),
        required=True,
        help="Specifies the node id of artificially partitioned datasets.",
    )
    args = parser.parse_args()
    partition_id = args.node_id

    # Load the partition data
    fds = FederatedDataset(dataset="hitorilabs/iris", partitioners={"train": N_CLIENTS})

    dataset = fds.load_partition(partition_id, "train")

    X = []
    # Use just the specified columns
    for column in column_names:
        X.append(dataset[column])

    # Start Flower client
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(X).to_client(),
    )
