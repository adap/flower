"""Handle the dataset partitioning and (optionally) complex downloads.

Please add here all the necessary logic to either download, uncompress, pre/post-process
your dataset (or all of the above). If the desired way of running your baseline is to
first download the dataset and partition it and then run the experiments, please
uncomment the lines below and tell us in the README.md (see the "Running the Experiment"
block) that this file should be executed first.
"""

import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset, random_split
from torchvision.datasets import CIFAR10, CIFAR100
from tqdm import tqdm

class Subset(Dataset):
    def __init__(self, data, labels, transform = None):
        self.data = torch.from_numpy(data)
        self.labels = torch.from_numpy(labels)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform != None:
            img = self.transform(img)
        return self.data[index], self.labels[index]

def _download_data(dataset_name: str) -> Tuple[Dataset, Dataset]:
    """Downloads (if necessary) and returns the CIFAR dataset.

    Parameters
    ----------
    dataset_name
        name of the torch dataset to be downloaded
    Returns
    -------
    Tuple[, CIFAR]
        The dataset for training and the dataset for testing CIFAR.
    """
    if dataset_name=="CIFAR10":
        train_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)),
                                    ])
        test_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)),
                                    ])
    else: # cifar100
        train_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                                    ])
        test_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                                    ])

    dataset = eval(dataset_name)
    trainset = dataset("./dataset", train=True, download=True, transform=train_transform)
    testset = dataset("./dataset", train=False, download=True, transform=test_transform)
    return trainset, testset


def _partition_data(
    num_clients,
    dataset_name: str,
    dirichlet: bool,
    dirichlet_coeff: int,
    n_active_categories: int,
    seed: Optional[int] = 21, # seed used for reproducing experiments
) -> Tuple[List[Dataset], Dataset]:
    
    """Split training set into iid or non iid partitions to simulate the
    federated setting.

    Parameters
    ----------
    num_clients : int
        Total number of participating clients 10% + 5% participating
    dataset_name: str
        name of the dataset to be used (torchvision)
    dirichlet: bool, optional
        Whether to follow a dirichlet distribution with replacement
        when assigning samples for each client, defaults to True
        Else use pathological partitioning
    dirichlet_coeff: int
        coefficient to be used for dirichlet distribution
    n_active_categories:
        number of categories to activate for a client
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[List[Dataset], Dataset]
        A list of dataset for each client and a single dataset to be use for testing the model.
    """
    
    trainset, testset = _download_data(dataset_name=dataset_name)
    n_classes = len(torch.unique(torch.tensor(trainset.targets)))

    debug = False
    if debug:
        print("############### using debug, iid partitioning #####################")
        partition_size = int(len(trainset) / num_clients)
        lengths = [partition_size] * num_clients
        datasets = random_split(trainset, lengths, torch.Generator().manual_seed(seed))
        return datasets, testset


    # load the entire dataset in 1 batch
    train_load = torch.utils.data.DataLoader(trainset, batch_size=50000, shuffle=False, num_workers=0)
    test_load = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=0) 

    # labels are of shape (n_data,)
    train_itr = train_load.__iter__(); test_itr = test_load.__iter__()
    train_x, train_y = train_itr.__next__()
    test_x, test_y = test_itr.__next__()

    train_x = train_x.numpy(); train_y = train_y.numpy().reshape(-1,1)
    test_x = test_x.numpy(); test_y = test_y.numpy().reshape(-1,1)
    
    # Shuffle Data
    # np.random.seed(seed)
    rand_perm = np.random.permutation(len(train_y))
    train_x = train_x[rand_perm]
    train_y = train_y[rand_perm]

    n_data_per_client = int((len(train_y)) / num_clients)
    client_data_list = np.ones(num_clients, dtype=int)*n_data_per_client
    diff = np.sum(client_data_list) - len(train_y)
    
    # Add/Subtract the excess number starting from first client
    if diff!= 0:
        for client_i in range(num_clients):
            if client_data_list[client_i] > diff:
                client_data_list[client_i] -= diff
                break

    if dirichlet:
        cls_priors = np.random.dirichlet(alpha=[dirichlet_coeff]*n_classes, size=num_clients)
        prior_cumsum = np.cumsum(cls_priors, axis=1)

    else: 
        c = n_active_categories
        a = np.ones([num_clients, n_classes])
        a[:,c::] = 0
        for i in a:
            np.random.shuffle(i)

        prior_cumsum = a.copy()
        for i in range(prior_cumsum.shape[0]):
            for j in range(prior_cumsum.shape[1]):
                if prior_cumsum[i,j] != 0:
                    prior_cumsum[i,j] = a[i,0:j+1].sum()/c*1.0
    
    # class wise index list in train_x, train_y
    idx_list = [np.where(train_y==i)[0] for i in range(n_classes)]
    cls_amount = [len(idx_list[i]) for i in range(n_classes)]
    true_sample = [0 for i in range(n_classes)]

    client_x = [np.zeros((client_data_list[client_idx], 3, 32, 32)).astype(np.float32) for client_idx in range(num_clients)]
    client_y = [np.zeros((client_data_list[client_idx], 1)).astype(np.int64) for client_idx in range(num_clients)]

    while(np.sum(client_data_list)!=0):
        curr_client = np.random.randint(num_clients)
        if client_data_list[curr_client] <= 0:
            continue

        client_data_list[curr_client] -= 1
        curr_prior = prior_cumsum[curr_client]

        while True:
            cls_label = np.argmax(np.random.uniform() <= curr_prior)
            if cls_amount[cls_label] <= 0:
                cls_amount [cls_label] = len(idx_list[cls_label]) 
                continue

            cls_amount[cls_label] -= 1
            true_sample[cls_label] += 1
            
            client_x[curr_client][client_data_list[curr_client]] = train_x[idx_list[cls_label][cls_amount[cls_label]]]
            client_y[curr_client][client_data_list[curr_client]] = train_y[idx_list[cls_label][cls_amount[cls_label]]]
            break
    
    # print classwise number of samples
    print(true_sample)
    client_x = np.asarray(client_x)
    client_y = np.asarray(client_y)

    datasets = []
    if dataset_name=="CIFAR10":
        train_transform = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)),
                                    ])
    else:
        train_transform = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                                    ])
        
    for i in range(num_clients):
        datasets.append(Subset(data=client_x[i], labels=client_y[i].flatten(), transform=train_transform))
    
    return datasets, testset
