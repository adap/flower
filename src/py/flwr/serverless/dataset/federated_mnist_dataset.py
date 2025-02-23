from dataclasses import dataclass


@dataclass
class FederatedMNISTDataset:
    """
    Under different partition methods, we can test the effectivness of a particular federated
    learning strategy.
    """
    partition_method: str = "by_class" # one of "random", "by_class"
    num_partitions: int = 2
    images = None  # (n, 28, 28)
    labels = None  # (n,)