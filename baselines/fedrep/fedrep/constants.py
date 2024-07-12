"""Constants used in machine learning pipeline."""

from enum import Enum


# FL Algorithms
class Algorithms(Enum):
    """Enum for FL algorithms."""

    FEDAVG = "FedAvg"
    FEDREP = "FedRep"


# FL Default Train and Fine-Tuning Epochs
DEFAULT_TRAIN_EP = 5
DEFAULT_FT_EP = 5

MEAN = {
    "cifar10": [0.485, 0.456, 0.406],
    "cifar100": [0.507, 0.487, 0.441],
}

STD = {
    "cifar10": [0.229, 0.224, 0.225],
    "cifar100": [0.267, 0.256, 0.276],
}
