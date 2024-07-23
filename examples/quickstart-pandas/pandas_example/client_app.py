"""pandas_example: A Flower / Pandas app."""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from flwr_datasets import FederatedDataset

from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context

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


def client_fn(context: Context) -> Client:
    """Construct a Client that will be run in a ClientApp.

    You can use settings in `context.run_config` to parameterize the
    construction of your Client. You could use the `context.node_config` to
    , for example, indicate which dataset to load (e.g accesing the partition-id).
    """

    # Read the node_config to fetch data partition associated to this node
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    # Load the partition data
    fds = FederatedDataset(
        dataset="hitorilabs/iris",
        partitioners={"train": num_partitions},
    )

    dataset = fds.load_partition(partition_id, "train").with_format("pandas")[:]
    # Use just the specified columns
    X = dataset[column_names]

    return FlowerClient(X).to_client()


app = ClientApp(client_fn=client_fn)
