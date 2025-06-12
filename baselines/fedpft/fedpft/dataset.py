"""fedpft: A Flower Baseline."""

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

FDS = None  # Cache FederatedDataset


def load_data(
    partition_id: int,
    num_partitions: int,
    dataset: str,
    batch_size: int,
    dirichlet_alpha: float,
    partition_by: str,
    image_column_name: str,
    transform: Compose,
    image_input_size: int,
    seed: int = 0,
    split_size: float = 0.8,
):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global FDS  # pylint: disable=global-statement
    if FDS is None:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by=partition_by,
            alpha=dirichlet_alpha,
            min_partition_size=10,
            self_balancing=True,
        )
        FDS = FederatedDataset(
            dataset=dataset,
            partitioners={"train": partitioner},
        )
    partition = FDS.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=split_size, seed=seed)

    def apply_transforms():
        """Apply transforms to the partition from FederatedDataset."""

        def batch_transform(batch):
            batch_img = [
                transform(img.resize((image_input_size, image_input_size)))
                for img in batch[image_column_name]
            ]
            batch_label = list(batch[partition_by])
            return {"img": batch_img, "label": batch_label}

        return batch_transform

    partition_train_test = partition_train_test.with_transform(apply_transforms())
    trainloader = DataLoader(partition_train_test["train"], batch_size=batch_size)
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader
