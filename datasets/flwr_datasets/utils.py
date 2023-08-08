"""Utils for FederatedDatasets."""
from typing import Dict

from flwr_datasets.partitioner import IidPartitioner, Partitioner


def _instantiate_partitioners(partitioners: Dict[str, int]) -> Dict[str, Partitioner]:
    """Transform the partitioners from the initial format to instantiated objects.

    Parameters
    ----------
    partitioners: Dict[str, int]
        Partitioners specified as split to the number of partitions format.

    Returns
    -------
    partitioners: Dict[str, Partitioner]
        Partitioners specified as split to Partitioner object.
    """
    instantiated_partitioners: Dict[str, Partitioner] = {}
    for key, value in partitioners.items():
        instantiated_partitioners[key] = IidPartitioner(num_partitions=value)
    return instantiated_partitioners


def _check_if_dataset_supported(dataset: str) -> None:
    """Check if the dataset is in the narrowed down list of the tested datasets."""
    if dataset not in ["mnist", "cifar10"]:
        raise ValueError(
            f"The currently tested and supported dataset are 'mnist' and "
            f"'cifar10'. Given: {dataset}"
        )
