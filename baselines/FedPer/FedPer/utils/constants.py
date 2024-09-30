from enum import Enum

# FL Algorithms
class Algorithms(Enum):
    FEDAVG = "FedAvg"
    FEDPER = "FedPer"

# FL Default Train and Fine-Tuning Epochs
DEFAULT_TRAIN_EP = 5
DEFAULT_FT_EP = 5