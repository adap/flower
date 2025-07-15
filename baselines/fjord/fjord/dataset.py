"""Dataset for CIFAR10."""

import random
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

CIFAR_NORMALIZATION = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))


class FLCifar10Client(Dataset):
    """Class implementing the partitioned CIFAR10 dataset."""

    def __init__(self, fl_dataset: Dataset, client_id: Optional[int] = None) -> None:
        """Ctor.

        Args:
        :param fl_dataset: The CIFAR10 dataset.
        :param client_id: The client id to be used.
        """
        self.fl_dataset = fl_dataset
        self.set_client(client_id)

    def set_client(self, index: Optional[int] = None) -> None:
        """Set the client to the given index. If index is None, use the whole dataset.

        Args:
        :param index: Index of the client to be used.
        """
        fl = self.fl_dataset
        if index is None:
            self.client_id = None
            self.length = len(fl.data)
            self.data = fl.data
        else:
            if index < 0 or index >= fl.num_clients:
                raise ValueError("Number of clients is out of bounds.")
            self.client_id = index
            indices = fl.partition[self.client_id]
            self.length = len(indices)
            self.data = fl.data[indices]
            self.targets = [fl.targets[i] for i in indices]

    def __getitem__(self, index: int):
        """Return the item at the given index.

        :param index: Index of the item to be returned.
        :return: The item at the given index.
        """
        fl = self.fl_dataset
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other fl_datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if fl.transform is not None:
            img = fl.transform(img)

        if fl.target_transform is not None:
            target = fl.target_transform(target)

        return img, target

    def __len__(self):
        """Return the length of the dataset."""
        return self.length


class FLCifar10(CIFAR10):
    """CIFAR10 Federated Dataset."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        root: str,
        train: Optional[bool] = True,
        transform: Optional[Module] = None,
        target_transform: Optional[Module] = None,
        download: Optional[bool] = False,
    ) -> None:
        """Ctor.

        :param root: Root directory of dataset
        :param train: If True, creates dataset from training set
        :param transform: A function/transform that takes in an PIL image and returns a
            transformed version.
        :param target_transform: A function/transform that takes in the target and
            transforms it.
        :param download: If true, downloads the dataset from the internet.
        """
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        # Uniform shuffle
        shuffle = np.arange(len(self.data))
        rng = np.random.default_rng(12345)
        rng.shuffle(shuffle)
        self.partition = shuffle.reshape([100, -1])
        self.num_clients = len(self.partition)


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Get the transforms for the CIFAR10 dataset.

    :return: The transforms for the CIFAR10 dataset.
    """
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*CIFAR_NORMALIZATION),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(*CIFAR_NORMALIZATION),
        ]
    )

    return transform_train, transform_test


def load_data(
    path: str, cid: int, train_bs: int, seed: int, eval_bs: int = 1024
) -> Tuple[DataLoader, DataLoader]:
    """Load the CIFAR10 dataset.

    :param path: The path to the dataset.
    :param cid: The client ID.
    :param train_bs: The batch size for training.
    :param seed: The seed to use for the random number generator.
    :param eval_bs: The batch size for evaluation.
    :return: The training and test sets.
    """

    def seed_worker(worker_id):  # pylint: disable=unused-argument
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)
    transform_train, transform_test = get_transforms()

    fl_dataset = FLCifar10(
        root=path, train=True, download=True, transform=transform_train
    )

    trainset = FLCifar10Client(fl_dataset, client_id=cid)
    testset = CIFAR10(root=path, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(
        trainset,
        batch_size=train_bs,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    test_loader = DataLoader(testset, batch_size=eval_bs)

    return train_loader, test_loader
