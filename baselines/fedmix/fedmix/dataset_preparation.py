"""..."""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10, CIFAR100


def _download_cifar10():
    """..."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = CIFAR10("./dataset", train=True, download=True, transform=transform)
    testset = CIFAR10("./dataset", train=False, download=True, transform=transform)
    return trainset, testset


def _download_cifar100():
    """..."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    trainset = CIFAR100("./dataset", train=True, download=True, transform=transform)
    testset = CIFAR100("./dataset", train=False, download=True, transform=transform)
    return trainset, testset


def _partition_trainset(
    trainset, num_classes, num_clients, num_classes_per_client, seed
):
    """..."""
    partition_size = int(len(trainset) / num_clients)
    np.random.seed(seed)

    class_indices = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(trainset):
        class_indices[label].append(idx)

    client_datasets = []

    for _client_id in range(num_clients):
        selected_classes = np.random.choice(
            num_classes, num_classes_per_client, replace=False
        )
        selected_indices = []

        for class_idx in selected_classes:
            selected_indices.extend(class_indices[class_idx])

        np.random.shuffle(selected_indices)
        client_indices = selected_indices[:partition_size]

        client_dataset = Subset(trainset, client_indices)
        client_datasets.append(client_dataset)

    return client_datasets


def _mash_data(client_datasets, mash_batch_size, num_classes):
    """..."""
    mashed_data = []
    for client_dataset in client_datasets:
        mashed_image, mashed_label = [], []
        for i, (image, label) in enumerate(client_dataset):
            mashed_image.append(image)
            mashed_label.append(torch.tensor([label]))
            if (i + 1) % mash_batch_size == 0:
                mashed_data.append(
                    (
                        torch.mean(torch.stack(mashed_image), dim=0),
                        torch.mean(
                            F.one_hot(
                                torch.squeeze(torch.stack(mashed_label)), num_classes
                            ).to(dtype=torch.float32),
                            dim=0,
                        ),
                    )
                )
                # print(mashed_data[0][1])
                mashed_image, mashed_label = [], []

    return mashed_data
