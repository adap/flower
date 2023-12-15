"""Dataset loading and processing utilities."""

import pickle
from typing import List, Tuple
import random 
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import omegaconf

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.targets = dataset.targets
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
    
def iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def noniid(dataset, no_participants, alpha=0.5):
    """
    Input: Number of participants and alpha (param for distribution)
    Output: A list of indices denoting data in CIFAR training set.
    Requires: cifar_classes, a preprocessed class-indice dictionary.
    Sample Method: take a uniformly sampled 10/100-dimension vector as parameters for
    dirichlet distribution to sample number of images in each class.
    """
    np.random.seed(666)
    random.seed(666)
    cifar_classes = {}
    for ind, x in enumerate(dataset):
        _, label = x
        if label in cifar_classes:
            cifar_classes[label].append(ind)
        else:
            cifar_classes[label] = [ind]

    per_participant_list = defaultdict(list)
    no_classes = len(cifar_classes.keys())
    class_size = len(cifar_classes[0])
    datasize = {}
    for n in range(no_classes):
        random.shuffle(cifar_classes[n])
        sampled_probabilities = class_size * np.random.dirichlet(
            np.array(no_participants * [alpha]))
        for user in range(no_participants):
            no_imgs = int(round(sampled_probabilities[user]))
            datasize[user, n] = no_imgs
            sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
            per_participant_list[user].extend(sampled_list)
            cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]
    train_img_size = np.zeros(no_participants)
    for i in range(no_participants):
        train_img_size[i] = sum([datasize[i,j] for j in range(no_classes)])
    clas_weight = np.zeros((no_participants,no_classes))
    for i in range(no_participants):
        for j in range(no_classes):
            clas_weight[i,j] = float(datasize[i,j])/float((train_img_size[i]))
    return per_participant_list, clas_weight


def load_datasets(
    config, num_clients, batch_size
) -> Tuple[List[DataLoader], DataLoader]:
    
    """Load the dataset and return the dataloaders for the clients and the server."""
    print("Loading data...")
    if config.name == "CIFAR10":
        Dataset = datasets.CIFAR10
    elif config.name == "CIFAR100":
        Dataset = datasets.CIFAR100
    else:
        raise NotImplementedError
    data_directory = f"./data/{config.name.lower()}/"
    ds_path = f"{data_directory}train_{num_clients}_{config.alpha:.2f}.pkl"
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    try:
        with open(ds_path, "rb") as file:
            train_datasets = pickle.load(file)
    except FileNotFoundError:
        dataset_train = Dataset(
            data_directory, train=True, download=True, transform=transform_train)
        if config.partition == "iid":
            train_datasets = iid(
                dataset_train,
                num_clients)
        else:
            train_datasets, _ = noniid(
                dataset_train,
                num_clients,
                config.alpha)
    dataset_test = Dataset(
        data_directory, train=False, download=True, transform=transform_test
    )
    test_loader = DataLoader(dataset_test, batch_size=batch_size, num_workers=2)
    train_loaders = [
        DataLoader(DatasetSplit(dataset_train, ids), batch_size=batch_size, shuffle=True, num_workers=2)
        for ids in train_datasets.values()
    ]

    return train_loaders, test_loader

