"""examplefkm: A Flower / Lifelines app."""

from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
from flwr.client import Client, ClientApp
from flwr.common import NDArray, NDArrays, Context

from examplefkm.task import load_partition


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


def client_fn(context: Context) -> Client:
    """Construct a Client that will be run in a ClientApp.

    You can use settings in `context.run_config` to parameterize the
    construction of your Client. You could use the `context.node_config` to, for
    example, indicate which dataset to load (e.g accesing the partition-id).
    """
    partition_id = context.node_config["partition-id"]
    times, events = load_partition(partition_id)
    return FlowerClient(times=times, events=events).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)
