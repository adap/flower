import argparse
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
from datasets import Dataset
from flwr.common import NDArray, NDArrays
from flwr_datasets.partitioner import NaturalIdPartitioner
from lifelines.datasets import load_waltons


class FlowerClient(fl.client.NumPyClient):
    """Flower client that holds and sends the events and times data.

    Parameters
    ----------
    times: NDArray
        Times of the `events`.
    events: NDArray
        Events represented by 0 - no event, 1 - event occurred.

    Raises
    ------
    ValueError
        If the `times` and `events` are not the same shape.
    """

    def __init__(self, times: NDArray, events: NDArray):
        if len(times) != len(events):
            raise ValueError("The times and events arrays have to be same shape.")
        self._times = times
        self._events = events

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[NDArrays, int, Dict]:
        return (
            [self._times, self._events],
            len(self._times),
            {},
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--node-id",
        type=int,
        required=True,
        help="Node id. Each node holds different part of the dataset.",
    )
    args = parser.parse_args()
    partition_id = args.node_id

    # Prepare data
    X = load_waltons()
    partitioner = NaturalIdPartitioner(partition_by="group")
    partitioner.dataset = Dataset.from_pandas(X)
    partition = partitioner.load_partition(partition_id).to_pandas()
    events = partition["E"].values
    times = partition["T"].values

    # Start Flower client
    client = FlowerClient(times=times, events=events).to_client()
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client,
    )
