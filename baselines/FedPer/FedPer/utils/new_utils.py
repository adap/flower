import json
from pathlib import Path
from typing import Any, Dict, Type, Union

from flwr.server.strategy import Strategy

from FedPer.utils.base_client import BaseClient
from FedPer.utils.constants import Algorithms
from FedPer.utils.FedPer_client import FedPerClient
from FedPer.utils.strategy_pipeline import (
    AggregateBodyStrategyPipeline,
    DefaultStrategyPipeline,
)


def load_config(config_file: Union[str, Path]) -> Dict[str, Any]:
    """Load the config file into a dictionary."""
    filepath = Path(config_file)

    f = open(filepath, mode="rt", encoding="utf-8")
    config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError(
            "config must be a dict or the name of a file containing a dict."
        )

    return config


def get_client_cls(algorithm: str) -> Type[BaseClient]:
    """Get client class from algorithm (default is FedAvg)."""
    if algorithm == Algorithms.FEDPER.value.lower():
        return FedPerClient
    elif algorithm == Algorithms.FEDAVG.value:
        return BaseClient
    else:
        raise ValueError(f"No such algorithm: {algorithm}")


def get_server_strategy(algorithm: str) -> Strategy:
    """Gets the server strategy pipeline corresponding to the received algorithm.

    Args:
        algorithm: the federated algorithm to be performed.

    Returns
    -------
        The pipeline to be used.
    """
    if algorithm == Algorithms.FEDPER.value:
        return AggregateBodyStrategyPipeline
    elif (
        algorithm == Algorithms.FEDAVG.value
    ):  # FedAvg, Proposal FedHybridAvgLG and Proposal FedHybridAvgLGDual
        return DefaultStrategyPipeline
    else:
        raise ValueError(f"No such algorithm: {algorithm}")
