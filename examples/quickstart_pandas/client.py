import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import flwr as fl


df = pd.read_csv("./data/client.csv")

column_names = ["sepal length (cm)", "sepal width (cm)"]


def compute_hist(df: pd.DataFrame, col_name: str) -> np.ndarray:
    vals, _ = np.histogram(df[col_name])
    return vals


# Define Flower client
class FlowerClient(fl.client.NumPyClient):

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        outputs = {}
        v_arr = []
        # Execute query locally
        for c in column_names:
            h = compute_hist(df, c)
            v_arr.append(h)
        return (
            v_arr,
            len(df),
            {},
        )


# Start Flower client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(),
)
