"""Digits dataset."""

import os
from random import shuffle
from typing import List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class DigitsDataset(Dataset):
    """Split datasets."""

    total_partitions: int = 10

    def __init__(  # pylint: disable=too-many-arguments
        self,
        data_path: str,
        channels: int,
        train: bool,
        partitions: Optional[List[int]] = None,
        transform=None,
    ):
        if train and partitions is not None:
            # Construct dataset by loading one or more partitions
            self.images, self.labels = np.load(
                os.path.join(
                    data_path,
                    f"partitions/train_part{partitions[0]}.pkl",
                ),
                allow_pickle=True,
            )
            for part in partitions[1:]:
                images, labels = np.load(
                    os.path.join(
                        data_path,
                        f"partitions/train_part{part}.pkl",
                    ),
                    allow_pickle=True,
                )
                self.images = np.concatenate([self.images, images], axis=0)
                self.labels = np.concatenate([self.labels, labels], axis=0)

        else:
            self.images, self.labels = np.load(
                os.path.join(data_path, "test.pkl"), allow_pickle=True
            )

        self.transform = transform
        self.channels = channels
        self.labels = self.labels.squeeze()

    def __len__(self) -> int:
        """Return number of images."""
        return self.images.shape[0]

    def __getitem__(self, idx):
        """Return a transformed example of the dataset."""
        image = self.images[idx]
        label = self.labels[idx]
        if self.channels == 1:
            image = Image.fromarray(image, mode="L")
        elif self.channels == 3:
            image = Image.fromarray(image, mode="RGB")
        else:
            raise ValueError(f"{self.channels} channel is not allowed.")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def load_partition(
    dataset: str, path_to_data: str, partition_indx: List[int], batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    """Load 'MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST_M' for the training and test.

    data to simulate a partition.
    """
    data_path = os.path.join(path_to_data, dataset)

    if dataset == "MNIST":
        transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        trainset = DigitsDataset(
            data_path=data_path,
            channels=1,
            partitions=partition_indx,
            train=True,
            transform=transform,
        )
        testset = DigitsDataset(
            data_path=data_path,
            channels=1,
            train=False,
            transform=transform,
        )

    elif dataset == "SVHN":
        transform = transforms.Compose(
            [
                transforms.Resize([28, 28]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        trainset = DigitsDataset(
            data_path=data_path,
            channels=3,
            partitions=partition_indx,
            train=True,
            transform=transform,
        )
        testset = DigitsDataset(
            data_path=data_path,
            channels=3,
            train=False,
            transform=transform,
        )

    elif dataset == "USPS":
        transform = transforms.Compose(
            [
                transforms.Resize([28, 28]),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        trainset = DigitsDataset(
            data_path=data_path,
            channels=1,
            partitions=partition_indx,
            train=True,
            transform=transform,
        )
        testset = DigitsDataset(
            data_path=data_path,
            channels=1,
            train=False,
            transform=transform,
        )

    elif dataset == "SynthDigits":
        transform = transforms.Compose(
            [
                transforms.Resize([28, 28]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        trainset = DigitsDataset(
            data_path=data_path,
            channels=3,
            partitions=partition_indx,
            train=True,
            transform=transform,
        )
        testset = DigitsDataset(
            data_path=data_path,
            channels=3,
            train=False,
            transform=transform,
        )

    elif dataset == "MNIST_M":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        trainset = DigitsDataset(
            data_path=data_path,
            channels=3,
            partitions=partition_indx,
            train=True,
            transform=transform,
        )
        testset = DigitsDataset(
            data_path=data_path,
            channels=3,
            train=False,
            transform=transform,
        )

    else:
        raise NotImplementedError(f"dataset: {dataset} is not available")

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader


# pylint: disable=too-many-locals
def get_data(dataset_cfg: DictConfig) -> List[Tuple[DataLoader, DataLoader, str]]:
    """Generate dataloaders for each client."""
    client_data = []
    d_cfg = dataset_cfg

    total_partitions = (
        10  # each dataset was pre-processed by the authors and split into 10 partitions
    )
    # First check that percent used is allowed
    allowed_percent = (np.arange(1, total_partitions + 1) / total_partitions).tolist()
    assert d_cfg.percent in allowed_percent, (
        f"'dataset.percent' should be in {allowed_percent}."
        "\nThis is because the trainset is pre-partitioned into 10 disjoint sets."
    )

    # Then check that with the percent selected, the desired number of clients (and
    # therefore dataloaders) can be created.
    max_expected_clients = len(d_cfg.to_include) * 1 / d_cfg.percent

    num_clients_step = len(d_cfg.to_include)
    possible_client_configs = np.arange(
        num_clients_step,
        max_expected_clients + num_clients_step,
        num_clients_step,
        dtype=np.int32,
    ).tolist()
    assert d_cfg.num_clients in possible_client_configs, (
        f"'dataset.num_clients' should be in {possible_client_configs}."
        "\n this is because you include {len(d_cfg.to_include)} datasets "
        f"(i.e. {d_cfg.to_include}) and each should be used by the same number"
        " of clients. The values of `num_clients` also depend on the"
        "'dataset.percent' you chose."
    )

    # All good, then create as many dataloaders as clients in the experiment.
    # Each dataloader might containe one or more partitions (depends on 'percent')
    # Each dataloader contains data of the same dataset.
    num_clients_per_dataset = d_cfg.num_clients // num_clients_step
    num_parts = int(d_cfg.percent * total_partitions)

    for dataset_name in dataset_cfg.to_include:
        parts = list(range(total_partitions))
        shuffle(parts)
        for i in range(num_clients_per_dataset):
            parts_for_client = parts[i * num_parts : (i + 1) * num_parts]
            print(f"{dataset_name = } | {parts_for_client = }")
            trainloader, testloader = load_partition(
                dataset_name,
                path_to_data=d_cfg.data_path,
                partition_indx=parts_for_client,
                batch_size=d_cfg.batch_size,
            )

            client_data.append((trainloader, testloader, dataset_name))

    # Ensure there is an entry in the list for each client
    assert (
        len(client_data) == d_cfg.num_clients
    ), f"{len(client_data) = } | {d_cfg.num_clients = }"

    return client_data
