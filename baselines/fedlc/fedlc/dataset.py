"""fedlc: A Flower Baseline."""

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from datasets import Dataset

FDS = None  # Cache FederatedDataset


def load_data(dataset: str, partition_id: int, num_partitions: int, batch_size: int, dirichlet_alpha: float):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global FDS  # pylint: disable=global-statement
    if FDS is None:
        dirichlet_partitioner = DirichletPartitioner(
            num_partitions=num_partitions, alpha=dirichlet_alpha, partition_by="label"
        )
        FDS = FederatedDataset(
            dataset=dataset,
            partitioners={"train": dirichlet_partitioner},
        )
    partition = FDS.load_partition(partition_id)
    
    # Remove last data 
    # Otherwise, with 20 clients a batch size might be 1,
    # which causes issues with batchnorm in resnet
    partition_dict = partition[:-1]
    partition = Dataset.from_dict(partition_dict)

    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    # cifar10 mean and std 
    pytorch_transforms = Compose(
        [ToTensor(), Normalize([0.491,0.482,0.446], [0.247,0.243,0.261])]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=batch_size, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader
