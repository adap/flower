from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from flwr.common import DataConnector


partition_id = 0

fds = FederatedDataset(dataset="cifar10", partitioners={"train": 2})
partition = fds.load_partition(partition_id)
# Divide data on each node: 80% train, 20% test
partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
pytorch_transforms = Compose(
    [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch

partition_train_test = partition_train_test.with_transform(apply_transforms)
trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
testloader = DataLoader(partition_train_test["test"], batch_size=32)


# DataConnector
dd = DataConnector({'train': trainloader, 'test': testloader})