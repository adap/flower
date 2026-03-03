import torch
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


_fds_cache: dict = {}

_transforms = {
    "mnist": Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]),
    "cifar10": Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
}

_img_keys = {
    "mnist": "image",
    "cifar10": "img",
}

_hf_names = {
    "mnist": "ylecun/mnist",
    "cifar10": "uoft-cs/cifar10",
}


def _make_transform_fn(dataset: str):
    """Return a transform function for the given dataset."""
    transform = _transforms[dataset]
    img_key = _img_keys[dataset]

    def apply_transforms(batch):
        batch[img_key] = [transform(img) for img in batch[img_key]]
        return batch

    return apply_transforms


def _make_collate_fn(dataset: str):
    """Return a collate function for the given dataset."""
    img_key = _img_keys[dataset]

    def collate_fn(batch):
        images = torch.stack([x[img_key] for x in batch])
        labels = torch.tensor([x["label"] for x in batch])
        return images, labels

    return collate_fn


def load_data(partition_id: int, num_partitions: int, dataset: str) -> tuple:
    """Load and partition data for a given client.

    Args:
        partition_id: Index of the partition to load.
        num_partitions: Total number of partitions.
        dataset: Dataset name, either 'mnist' or 'cifar10'.

    Returns:
        Tuple of (trainloader, testloader).
    """
    if dataset not in _hf_names:
        raise ValueError(f"Unsupported dataset: '{dataset}'. Choose 'mnist' or 'cifar10'.")

    if dataset not in _fds_cache:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        _fds_cache[dataset] = FederatedDataset(
            dataset=_hf_names[dataset],
            partitioners={"train": partitioner},
        )

    partition = _fds_cache[dataset].load_partition(partition_id)
    partition = partition.train_test_split(test_size=0.2, seed=42)
    partition = partition.with_transform(_make_transform_fn(dataset))

    collate_fn = _make_collate_fn(dataset)

    trainloader = DataLoader(
        partition["train"], batch_size=32, shuffle=True, collate_fn=collate_fn
    )
    testloader = DataLoader(
        partition["test"], batch_size=32, collate_fn=collate_fn
    )

    return trainloader, testloader
