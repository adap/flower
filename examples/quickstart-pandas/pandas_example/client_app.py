"""pandas_example: A Flower / Pandas app."""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from flwr_datasets import FederatedDataset

from flwr.client import Client, ClientApp, NumPyClient

column_names = ["sepal_length", "sepal_width"]


def compute_hist(df: pd.DataFrame, col_name: str) -> np.ndarray:
    freqs, _ = np.histogram(df[col_name])
    return freqs


# Define Flower client
class FlowerClient(NumPyClient):
    def __init__(self, X: pd.DataFrame):
        self.X = X

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        hist_list = []
        # Execute query locally
        for c in self.X.columns:
            hist = compute_hist(self.X, c)
            hist_list.append(hist)
        return (
            hist_list,
            len(self.X),
            {},
        )


N_CLIENTS = 2


def client_fn(node_id, partition_id) -> Client:
    """Client function to return an instance of Client()."""
    # Load the partition data
    fds = FederatedDataset(dataset="hitorilabs/iris", partitioners={"train": N_CLIENTS})

    dataset = fds.load_partition(partition_id, "train").with_format("pandas")[:]
    # Use just the specified columns
    X = dataset[column_names]

    return FlowerClient(X).to_client()


app = ClientApp(client_fn=client_fn)
