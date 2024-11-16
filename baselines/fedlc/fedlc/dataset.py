"""fedlc: A Flower Baseline."""

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip

from flwr.common import Context

FDS = None  # Cache FederatedDataset


def get_data_transforms(dataset: str):
    if dataset == "cifar10":
        tfms = Compose(
            [
                RandomCrop(32, padding=4),
                RandomHorizontalFlip(),
                ToTensor(), 
                Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ]
        )
    else:
        raise ValueError("Only cifar10 is supported!")
    return tfms


def get_transforms_apply_fn(transforms, partition_by):
    def apply_transforms(batch):
        batch["img"] = [transforms(img) for img in batch["img"]]
        batch["label"] = batch[partition_by]
        return batch

    return apply_transforms


def get_transformed_ds(ds, dataset_name, partition_by) -> Dataset:
    tfms = get_data_transforms(dataset_name)
    transform_fn = get_transforms_apply_fn(tfms, partition_by)
    return ds.with_transform(transform_fn)


def load_data(context: Context):
    """Load partitioned data for clients.

    Only used for client-side training.
    """
    dirichlet_alpha = float(context.run_config["dirichlet-alpha"])
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    dataset = str(context.run_config["dataset"])
    batch_size = int(context.run_config["batch-size"])
    partition_by = str(context.run_config["dataset-partition-by"])

    # Only initialize `FederatedDataset` once
    global FDS  # pylint: disable=global-statement

    if FDS is None:
        dirichlet_partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            alpha=dirichlet_alpha,
            partition_by=partition_by,
            min_partition_size=10,
        )
        FDS = FederatedDataset(
            dataset=dataset,
            partitioners={"train": dirichlet_partitioner},
        )

    train_partition = FDS.load_partition(partition_id)
    train_partition.set_format("torch")

    trainloader = DataLoader(
        get_transformed_ds(train_partition, dataset, partition_by),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    return trainloader, train_partition["label"]
