"""fedrep: A Flower Baseline."""

from enum import Enum

DEFAULT_LOCAL_TRAIN_EPOCHS: int = 10
DEFAULT_FINETUNE_EPOCHS: int = 5
DEFAULT_REPRESENTATION_EPOCHS: int = 1
FEDREP_HEAD_STATE = "fedrep_head_state"

MEAN = {"cifar10": [0.485, 0.456, 0.406], "cifar100": [0.507, 0.487, 0.441]}

STD = {"cifar10": [0.229, 0.224, 0.225], "cifar100": [0.267, 0.256, 0.276]}


class Algorithm(Enum):
    """Algorithm names."""

    FEDREP = "fedrep"
    FEDAVG = "fedavg"


class ModelDatasetName(Enum):
    """Dataset names."""

    CNN_CIFAR_10 = "cnncifar10"
    CNN_CIFAR_100 = "cnncifar100"
