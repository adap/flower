import torch
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST


def get_mnist(data_path: str = "./data"):
    """Downlaod MNIST and apply a simple transform."""

    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    trainset = MNIST(data_path, train=True, download=True, transform=tr)
    testset = MNIST(data_path, train=False, download=True, transform=tr)

    return trainset, testset


def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float = 0.1):
    """Download and partition the MNIST dataset."""

    # We are not doing any Hydra magic here but this is one place you'd normally
    # add some of it. Why? because you probably want different datasets in your
    # Flower experiments. Each could easily require a different partitioning mechanism
    # or none at all. For example, if you want to use MNIST, CIFAR-10, FEMNIST, SpeechCommands,
    # these require very different loading/partitioning/preprocessing methodologies.
    # having this arranged via Hydra configs might be a small upfront cost but it pays off.

    trainset, testset = get_mnist()

    # split trainset into `num_partitions` trainsets
    num_images = len(trainset) // num_partitions

    partition_len = [num_images] * num_partitions

    trainsets = random_split(
        trainset, partition_len, torch.Generator().manual_seed(2023)
    )

    # create dataloaders with train+val support
    trainloaders = []
    valloaders = []
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(
            trainset_, [num_train, num_val], torch.Generator().manual_seed(2023)
        )

        trainloaders.append(
            DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2)
        )
        valloaders.append(
            DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2)
        )

    testloader = DataLoader(testset, batch_size=128)

    return trainloaders, valloaders, testloader
