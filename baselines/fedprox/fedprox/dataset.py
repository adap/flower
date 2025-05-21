"""fedprox: A Flower Baseline."""

import numpy as np
from datasets import DatasetDict, load_dataset
from easydict import EasyDict
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DistributionPartitioner
from flwr_datasets.preprocessor import Preprocessor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

FDS = None  # Cache FederatedDataset

MNIST_TRANSFORMS = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])


class FEMNISTFilter(Preprocessor):
    """A Preprocessor class that filter the FEMNIST data.

    It filters data with label 0 to 9 (lower case letters 'a'-'j')
    """

    def __call__(self, dataset: DatasetDict) -> DatasetDict:
        """."""
        allowed_labels = list(range(10))  # mapping to 'a'-'j'
        filtered_dataset = dataset.filter(
            lambda example: example["character"] in allowed_labels
        )
        return filtered_dataset


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["image"] = [MNIST_TRANSFORMS(img) for img in batch["image"]]

    return batch


def process_femnist(dataset):
    """Process FEMNIST when setting up centralised test data."""
    return dataset.filter(lambda example: example["character"] in list(range(10)))


def load_data(
    dataset_config: EasyDict,
    partition_id: int,
    num_partitions: int,
):
    """Load and partition data."""
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
        label_key = "character" if "femnist" in dataset_config.path else "label"
        partitioner = DistributionPartitioner(
            distribution_array=distribution_array,
            num_partitions=num_partitions,
            num_unique_labels_per_partition=labels_per_partition,
            partition_by=label_key,  # target column `label` ("character" for FEMNIST)
            preassigned_num_samples_per_label=samples_per_label,
        )
        if "femnist" in dataset_config.path:
            FDS = FederatedDataset(
                dataset=dataset_config.path,
                partitioners={"train": partitioner},
                preprocessor=FEMNISTFilter(),  # Add the Preprocessor class for FEMNIST
            )
        else:
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
    partition_train_test = partition_train_test.with_transform(apply_transforms)
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
    """Generate the dataloader for the test set.

    Args:
        dataset_config (dict): The dataset configuration.

    Note: FEMNIST does not have a test data, so we need to manually process the
    training data to create test data.

    Returns
    -------
        DataLoader: The MNIST test set dataloader.
    """
    if "femnist" in dataset_config.path:
        dataset = load_dataset(path=dataset_config.path)["train"]
        split_dataset = dataset.train_test_split(
            test_size=dataset_config.val_ratio, seed=dataset_config.seed
        )
        test_dataset = process_femnist(split_dataset["test"])
        test_dataset = test_dataset.with_transform(apply_transforms)
    else:
        test_dataset = load_dataset(path=dataset_config.path)["test"].with_transform(
            apply_transforms
        )
    return DataLoader(test_dataset, batch_size=dataset_config.batch_size)
