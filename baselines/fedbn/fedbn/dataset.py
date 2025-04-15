"""fedbn: A Flower Baseline."""

import os
from pathlib import Path
from random import shuffle
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from flwr.common import Context

DATA_DIRECTORY = Path(os.path.abspath(__file__)).parent.parent / "data"


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
    """Load 'MNIST','SVHN', 'USPS', 'SynthDigits', 'MNIST_M' data.

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
def get_data(context: Context) -> List[Tuple[DataLoader, DataLoader, str]]:
    """Generate dataloaders for each client."""
    client_data = []
    run_config = context.run_config
    to_include = str(run_config["to-include"]).split(",")
    percent = float(run_config["percent"])
    num_clients = int(run_config["num-clients"])

    total_partitions = 10
    # each dataset was pre-processed by the authors and split
    # into 10 partitions. First check that percent
    # used is allowed
    allowed_percent = np.arange(1, total_partitions + 1) / total_partitions
    message = (
        f"'dataset.percent' should be in {list(allowed_percent)}."
        "\nTrainset is pre-partitioned into 10 disjoint sets."
    )
    assert percent in allowed_percent, message
    # Check that with the percent selected, the desired number of clients (and
    # therefore dataloaders) can be created.
    max_expected_clients = len(to_include) * 1 / percent

    num_clients_step = len(to_include)
    possible_client_configs = np.arange(
        num_clients_step,
        max_expected_clients + num_clients_step,
        num_clients_step,
        dtype=np.int32,
    ).tolist()
    assert num_clients in possible_client_configs, (
        f"'num-clients' should be in {possible_client_configs}."
        f"\n this is because you include {to_include} datasets "
        f"(i.e. {to_include} and each should be used by the same number"
        " of clients. The values of `num_clients` also depend on the"
        "'dataset.percent' you chose."
    )

    # All good, then create as many dataloaders as clients in the experiment.
    # Each dataloader might contain 1+ partitions (depends on 'percent')
    # Each dataloader contains data of the same dataset.
    num_clients_per_dataset = int(num_clients // num_clients_step)
    num_parts = int(percent * total_partitions)

    for dataset_name in to_include:
        parts = list(range(total_partitions))
        shuffle(parts)
        for i in range(num_clients_per_dataset):
            front = i * num_parts
            back = (i + 1) * num_parts
            parts_for_client = parts[front:back]
            # print(f"{dataset_name = } | {parts_for_client = }")
            trainloader, testloader = load_partition(
                dataset_name,
                path_to_data=str(DATA_DIRECTORY),
                partition_indx=parts_for_client,
                batch_size=int(run_config["batch-size"]),
            )

            client_data.append((trainloader, testloader, dataset_name))
    # Ensure there is an entry in the list for each client
    assert (
        len(client_data) == num_clients
    ), f"{len(client_data) = } | {num_clients = }"

    return client_data
