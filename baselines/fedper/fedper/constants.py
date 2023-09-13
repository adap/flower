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
}

STD = {
    "cifar10": [0.2023, 0.1994, 0.201],
}
