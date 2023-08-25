from dataset import load_datasets
from omegaconf import DictConfig

config = DictConfig({
    "iid": True,
    "balance": False,
    "power_law": False,
    "datasets": ["cifar10"],
})

# Get data loaders
trainloaders, valloaders, testloaders = load_datasets(
    config=config, 
    num_clients=3
)

