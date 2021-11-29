import numpy as np
import torch
import torchvision

from flwr.dataset.utils.common import create_lda_partitions
from flwr.dataset.utils.common import XY
from typing import List, Tuple
from torchvision.datasets import CIFAR10

list_partitions, dist = create_lda_partitions(
    XY=dataset, num_partitions=num_partitions, concentration=concentration
)


def save_partitions(partition_dir: Path, list_partitions: XYList):
    partition_dir.mkdir(parents=True, exist_ok=True)
    for idx, partition in enumerate(list_partitions):
        partition_file = partition_dir / f"{idx:03d}.pickle"
        with open(partition_file, "rb") as f:
            pickle.dump(partition, f)


def generate_partitions(root: str = "./data"):
    for is_train, dataset in [(True, "train"), (False, "test")]:
        list_partitions = partition_cifar10(root=root, train=is_train)
        partition_dir = Path(root) / dataset
        save_partitions(partition_dir=partition_dir, list_partitions=list_partitions)
