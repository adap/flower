"""fedrep: A Flower Baseline."""

from typing import Tuple

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import PathologicalPartitioner
from flwr_datasets.preprocessor import Merger
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from flwr.common import Context

from .constants import MEAN, STD

FDS = None  # Cache FederatedDataset


def load_data(
    partition_id: int, num_partitions: int, context: Context
) -> Tuple[DataLoader, DataLoader]:
    """Split the data and return training and testing data for the specified partition.

    Parameters
    ----------
    partition_id : int
        Partition number for which to retrieve the corresponding data.
    num_partitions : int
        Total number of partitions.
    context: Context
        the context of the current run.

    Returns
    -------
    data : Tuple[DataLoader, DataLoader]
        A tuple with the training and testing data for the current partition_id.
    """
    batch_size = int(context.run_config["batch-size"])
    dataset_name = str(context.run_config["dataset-name"]).lower()
    dataset_split_num_classes = int(context.run_config["dataset-split-num-classes"])
    dataset_split_seed = int(context.run_config["dataset-split-seed"])
    dataset_split_fraction = float(context.run_config["dataset-split-fraction"])

    # - you can define your own data transformation strategy here -
    # These transformations are from the official repo
    train_data_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN[dataset_name], STD[dataset_name]),
        ]
    )
    test_data_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN[dataset_name], STD[dataset_name]),
        ]
    )

    use_fine_label = False
    if dataset_name == "cifar100":
        use_fine_label = True

    partitioner = PathologicalPartitioner(
        num_partitions=num_partitions,
        partition_by="fine_label" if use_fine_label else "label",
        num_classes_per_partition=dataset_split_num_classes,
        class_assignment_mode="random",
        shuffle=True,
        seed=dataset_split_seed,
    )

    global FDS  # pylint: disable=global-statement
    if FDS is None:
        FDS = FederatedDataset(
            dataset=dataset_name,
            partitioners={"all": partitioner},
            preprocessor=Merger({"all": ("train", "test")}),
        )

    def apply_train_transforms(batch: Dataset) -> Dataset:
        """Apply transforms for train data to the partition from FederatedDataset."""
        batch["img"] = [train_data_transform(img) for img in batch["img"]]
        if use_fine_label:
            batch["label"] = batch["fine_label"]
        return batch

    def apply_test_transforms(batch: Dataset) -> Dataset:
        """Apply transforms for test data to the partition from FederatedDataset."""
        batch["img"] = [test_data_transform(img) for img in batch["img"]]
        if use_fine_label:
            batch["label"] = batch["fine_label"]
        return batch

    partition = FDS.load_partition(partition_id, split="all")

    partition_train_test = partition.train_test_split(
        train_size=dataset_split_fraction, shuffle=True, seed=dataset_split_seed
    )

    trainset = partition_train_test["train"].with_transform(apply_train_transforms)
    testset = partition_train_test["test"].with_transform(apply_test_transforms)

    trainloader = DataLoader(trainset, batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size)

    return trainloader, testloader
