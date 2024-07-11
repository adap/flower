import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import flwr as fl

class FileNotFoundErrorWithDirContents(Exception):
    def __init__(self, message):
        super().__init__(message)

def read_csv_with_error_handling(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError as e:
        directory = os.path.dirname(file_path)
        try:
            contents = os.listdir(directory)
            raise FileNotFoundErrorWithDirContents(
                f"{str(e)}\nDirectory contents: {', '.join(contents)}"
            ) from e
        except Exception as dir_error:
            raise FileNotFoundErrorWithDirContents(
                f"{str(e)}\nCould not list directory contents due to: {str(dir_error)}"
            ) from e

# Example usage
try:
    df = read_csv_with_error_handling("./framework-pandas/data/client.csv")
except FileNotFoundErrorWithDirContents as e:
    print(e)

column_names = ["sepal length (cm)", "sepal width (cm)"]


def compute_hist(df: pd.DataFrame, col_name: str) -> np.ndarray:
    freqs, _ = np.histogram(df[col_name])
    return freqs


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
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


def client_fn(cid):
    for entry in os.listdir("."):
        print(entry)
    return FlowerClient().to_client()


app = fl.client.ClientApp(
    client_fn=client_fn,
)

if __name__ == "__main__":
    # Start Flower client
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient().to_client(),
    )
