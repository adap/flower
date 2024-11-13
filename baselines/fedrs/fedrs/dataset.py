"""fedrs: A Flower Baseline."""

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import PathologicalPartitioner
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)

from flwr.common import Context

FDS = None  # Cache FederatedDataset


def get_data_transforms(dataset: str, split: str):
    """Return data transforms for dataset."""
    if dataset != "cifar10":
        raise ValueError("Only cifar10 is supported!")

    tfms = []
    if split == "train":
        tfms = [
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
        ]

    tfms.extend(
        [
            ToTensor(),
            Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ]
    )
    return Compose(tfms)


def _get_transforms_apply_fn(transforms, partition_by):
    def apply_transforms(batch):
        batch["img"] = [transforms(img) for img in batch["img"]]
        batch["label"] = batch[partition_by]
        return batch

    return apply_transforms


def get_transformed_ds(ds, dataset_name, partition_by, split) -> Dataset:
    """Return dataset with transformations applied."""
    tfms = get_data_transforms(dataset_name, split)
    transform_fn = _get_transforms_apply_fn(tfms, partition_by)
    return ds.with_transform(transform_fn)


def load_data(context: Context):
    """Load partitioned data for clients.

    Only used for client-side training.
    """
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    dataset = str(context.run_config["dataset"])
    batch_size = int(context.run_config["batch-size"])
    partition_by = str(context.run_config["dataset-partition-by"])
    num_classes_per_partition = int(context.run_config["num-classes-per-partition"])

    # Only initialize `FederatedDataset` once
    global FDS  # pylint: disable=global-statement

    if FDS is None:
        partitioner = PathologicalPartitioner(
            num_partitions=num_partitions,
            partition_by=partition_by,
            num_classes_per_partition=num_classes_per_partition,
            shuffle=True,
        )
        FDS = FederatedDataset(
            dataset=dataset,
            partitioners={"train": partitioner},
        )

    train_partition = FDS.load_partition(partition_id)
    train_partition.set_format("torch")

    trainloader = DataLoader(
        get_transformed_ds(train_partition, dataset, partition_by, split="train"),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    return trainloader, train_partition["label"]
