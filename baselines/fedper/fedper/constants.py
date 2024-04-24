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
    "cifar10": [0.4915, 0.4823, 0.4468],
}

STD = {
    "cifar10": [0.2470, 0.2435, 0.2616],
}
