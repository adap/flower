from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from flwr.client import ClientApp, NumPyClient, start_client
from flwr.common import Context

absolute_path = str(Path(".").absolute())
try:
    df = pd.read_csv("./data/client.csv")
except FileNotFoundError:
    df = pd.read_csv(f"{absolute_path}/data/client.csv")

column_names = ["sepal length (cm)", "sepal width (cm)"]


def compute_hist(df: pd.DataFrame, col_name: str) -> np.ndarray:
    freqs, _ = np.histogram(df[col_name])
    return freqs


# Define Flower client
class FlowerClient(NumPyClient):
    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        hist_list = []
        # Execute query locally
        for c in column_names:
            hist = compute_hist(df, c)
            hist_list.append(hist)
        return (
            hist_list,
            len(df),
            {},
        )


def client_fn(context: Context):
    return FlowerClient().to_client()


app = ClientApp(
    client_fn=client_fn,
)

if __name__ == "__main__":
    # Start Flower client
    start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient().to_client(),
    )
