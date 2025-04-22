"""fedprox: A Flower Baseline."""

import numpy as np
from datasets import load_dataset
from easydict import EasyDict
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DistributionPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

FDS = None  # Cache FederatedDataset

MNIST_TRANSFORMS = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
CIFAR_TRANSFORMS = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def apply_transforms(batch, path):
    """Apply transforms to the partition from FederatedDataset."""
    if "mnist" in path:
        batch["image"] = [MNIST_TRANSFORMS(img) for img in batch["image"]]
    elif "cifar" in path:
        batch["img"] = [CIFAR_TRANSFORMS(img) for img in batch["img"]]

    return batch


def load_data(
    dataset_config: EasyDict,
    partition_id: int,
    num_partitions: int,
):
    """Load and partition MNIST or CIFAR data."""
    # Only initialize `FederatedDataset` once
    global FDS  # pylint: disable=global-statement
    if FDS is None:
        # Generate a vector from a log-normal probability distribution
        rng = np.random.default_rng(dataset_config.seed)
        distribution_array = rng.lognormal(
            dataset_config.mu,
            dataset_config.sigma,
            (num_partitions * dataset_config.num_unique_labels_per_partition),
        )
        distribution_array = distribution_array.reshape(
            (dataset_config.num_unique_labels, -1)
        )
        labels_per_partition = dataset_config.num_unique_labels_per_partition
        samples_per_label = dataset_config.preassigned_num_samples_per_label
        partitioner = DistributionPartitioner(
            distribution_array=distribution_array,
            num_partitions=num_partitions,
            num_unique_labels_per_partition=labels_per_partition,
            partition_by="label",  # MNIST dataset has a target column `label`
            preassigned_num_samples_per_label=samples_per_label,
        )
        FDS = FederatedDataset(
            dataset=dataset_config.path,
            partitioners={"train": partitioner},
        )

    partition = FDS.load_partition(partition_id)
    # Divide data on each node: 90% train, 10% test
    partition_train_test = partition.train_test_split(
        test_size=dataset_config.val_ratio, seed=dataset_config.seed
    )
    # The validation set is never used because we do centralized evaluation
    # on the server on the held-out test dataset.
    partition_train_test = partition_train_test.with_transform(lambda batch: apply_transforms(batch, path=dataset_config.path))
    return (
        DataLoader(
            partition_train_test["train"],
            batch_size=dataset_config.batch_size,
            shuffle=True,
        ),
        DataLoader(
            partition_train_test["test"],
            batch_size=dataset_config.batch_size,
        ),
    )


def prepare_test_loader(dataset_config: EasyDict):
    """Generate the dataloader for the MNIST test set.

    Args:
        dataset_config (dict): The dataset configuration.

    Returns
    -------
        DataLoader: The MNIST test set dataloader.
    """
    test_dataset = load_dataset(path=dataset_config.path)["test"].with_transform(
        lambda batch: apply_transforms(batch, path=dataset_config.path)
    )
    return DataLoader(test_dataset, batch_size=dataset_config.batch_size)
