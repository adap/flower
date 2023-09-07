"""Constants used in machine learning pipeline."""
from enum import Enum


# FL Algorithms
class Algorithms(Enum):
    """Enum for FL algorithms."""

    FEDAVG = "FedAvg"
    FEDPER = "FedPer"


# FL Default Train and Fine-Tuning Epochs
DEFAULT_TRAIN_EP = 5
DEFAULT_FT_EP = 5

MEAN = {
    "cifar10": [0.4914, 0.4822, 0.4465],
    "cifar100": [0.5071, 0.4865, 0.4409],
}

STD = {
    "cifar10": [0.2023, 0.1994, 0.201],
    "cifar100": [0.2009, 0.1984, 0.2023],
}
