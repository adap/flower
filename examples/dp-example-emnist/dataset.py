import torch

from torch.utils.data import random_split, DataLoader
from torchvision.datasets import EMNIST
from torchvision.transforms import ToTensor, Normalize, Compose

def get_emnist(data_path: str = './data'):

    trainset = EMNIST(data_path, split='digits', train=True, download=True, transform=ToTensor())
    testset = EMNIST(data_path, split='digits', train=False, download=True, transform=ToTensor())

    return trainset, testset


def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float = 0.1):
    trainset, testset = get_emnist()

    num_images = len(trainset) // num_partitions

    partition_len = [num_images] * num_partitions

    trainsets = random_split (trainset, partition_len, torch.Generator().manual_seed(2023))


    trainloaders = []
    valloaders = []
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(trainset_, [num_train, num_val], torch.Generator().manual_seed(2023))

        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2))


    testloader = DataLoader(testset, batch_size=128)


    return trainloaders, valloaders, testloader

