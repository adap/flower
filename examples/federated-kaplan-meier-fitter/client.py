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


# Prepare data
X = load_waltons()
partitioner = NaturalIdPartitioner(partition_by="group")
partitioner.dataset = Dataset.from_pandas(X)


def get_client_fn(partition_id: int):
    def client_fn(cid: str):
        partition = partitioner.load_partition(partition_id).to_pandas()
        events = partition["E"].values
        times = partition["T"].values
        return FlowerClient(times=times, events=events).to_client()

    return client_fn


# Run via `flower-client-app client:app`
node_1_app = fl.client.ClientApp(
    client_fn=get_client_fn(0),
)
node_2_app = fl.client.ClientApp(
    client_fn=get_client_fn(1),
)
