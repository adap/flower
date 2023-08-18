"""Handle the dataset partitioning and (optionally) complex downloads.

Please add here all the necessary logic to either download, uncompress, pre/post-process
your dataset (or all of the above). If the desired way of running your baseline is to
first download the dataset and partition it and then run the experiments, please
uncomment the lines below and tell us in the README.md (see the "Running the Experiment"
block) that this file should be executed first.
"""
# import hydra
# from hydra.core.hydra_config import HydraConfig
# from hydra.utils import call, instantiate
# from omegaconf import DictConfig, OmegaConf


# @hydra.main(config_path="conf", config_name="base", version_base=None)
# def download_and_preprocess(cfg: DictConfig) -> None:
#     """Does everything needed to get the dataset.

#     Parameters
#     ----------
#     cfg : DictConfig
#         An omegaconf object that stores the hydra config.
#     """

#     ## 1. print parsed config
#     print(OmegaConf.to_yaml(cfg))

#     # Please include here all the logic
#     # Please use the Hydra config style as much as possible specially
#     # for parts that can be customised (e.g. how data is partitioned)

# if __name__ == "__main__":

#     download_and_preprocess()

from typing import List, Optional, Tuple

import os
import wget
import numpy as np
import torch
import random
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from zipfile import ZipFile
from torch.utils.data import ConcatDataset, Dataset, Subset, random_split
from torchvision.datasets import CIFAR10

def _download_data(dataset: str = 'cifar10') -> Tuple[Dataset, Dataset]:
    """Downloads (if necessary) and returns the MNIST dataset.

    Returns
    -------
    Tuple[Dataset, Dataset]
        The training and testing datasets.
    """
    if dataset == "cifar10":
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = CIFAR10(
            root="./dataset/cifar10", train=True, download=True, transform=transform
        )
        testset = CIFAR10(
            root="./dataset/cifar10", train=False, download=True, transform=transform
        )
        return trainset, testset
    elif dataset == "FLICKR-AES":
        raise NotImplementedError("FLICKR-AES dataset not implemented.")
        # Get zip file from 
        # https://drive.google.com/file/d/1jY7GMMNaQGQ80AAL99FLrBpWPFCwTKVT/view?usp=drive_link
        if not os.path.exists("./dataset/FLICKR-AES"):
            os.makedirs("./dataset/FLICKR-AES")
        if not os.path.exists("./dataset/FLICKR-AES/FLICKR-AES.zip"):
            wget.download(
                "https://drive.google.com/u/0/uc?id=1jY7GMMNaQGQ80AAL99FLrBpWPFCwTKVT&export=download",
                "./dataset/FLICKR-AES/FLICKR-AES.zip",
            )
        with ZipFile("./dataset/FLICKR-AES/FLICKR-AES.zip", "r") as zipObj:
            zipObj.extractall("./dataset/FLICKR-AES")

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        trainset = ImageFolder(
            "./dataset/FLICKR-AES/FLICKR-AES/train", transform=transform
        )
        testset = ImageFolder(
            "./dataset/FLICKR-AES/FLICKR-AES/test", transform=transform
        )
        return trainset, testset

    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented.")


def _partition_data(
    num_clients,
    dataset_name: str,
    iid: Optional[bool] = False,
    num_classes : Optional[int] = 10,
    seed: Optional[int] = 42,
) -> Tuple[List[Dataset], Dataset]:
    """Split training set into iid or non iid partitions to simulate the
    federated setting.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    dataset : str
        The name of the dataset to be used
    iid : bool, optional
        Whether the data should be independent and identically distributed between
        the clients or if the data should first be sorted by labels and distributed by chunks
        to each client (used to test the convergence in a worst case scenario), by default False
    num_classes : int, optional
        The number of classes in the dataset, by default 10
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[List[Dataset], Dataset]
        A list of dataset for each client and a single dataset to be use for testing the model.
    """

    # Download dataset
    trainset, testset = _download_data(dataset=dataset_name)

    # Get number of classes in the dataset
    total_num_classes = _get_num_classes(dataset_name)
    
    # Partition data
    lengths = _get_split_lenghts(trainset, num_clients)

    # Get datasets 
    datasets, testset = get_datasets(
        trainset, testset, 
        num_clients, total_num_classes, 
        lengths, iid, num_classes, seed
    )

    return datasets, testset

    
def _get_num_classes(dataset_name: str) -> int:
    """Returns the number of classes in the dataset.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to be used

    Returns
    -------
    int
        The number of classes in the dataset.
    """
    if dataset_name == "cifar10":
        return 10
    elif dataset_name == "FLICKR-AES":
        raise NotImplementedError("FLICKR-AES dataset not implemented.")
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented.")


def _get_split_lenghts(dataset: Dataset, num_clients: int, ) -> List[int]:
    """Returns a list with the length of each partition.

    Parameters
    ----------
    dataset : Dataset
        The dataset to be partitioned
    num_clients : int
        The number of clients that hold a part of the data
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    List[int]
        A list with the length of each partition.
    """
    partition_size = int(len(dataset) / num_clients)
    lengths = [partition_size] * num_clients
    if sum(lengths) < len(dataset):
        remaining_data = len(dataset) - sum(lengths)
        lengths[-1] += remaining_data
    return lengths

def get_datasets(
        trainset : Dataset, testset: Dataset, num_clients: int, total_num_classes: int, lengths : List[int],
        iid : bool = False, num_classes : int = 10, seed : int = 42, ):
    """
        Split training set into iid or non iid partitions to simulate the federated setting.
        
        Parameters
        ----------
        num_clients : int
            The number of clients that hold a part of the dataset. 
        trainset : Dataset
            The dataset to be partitioned
        testset : Dataset
            The dataset to be used for testing
        iid : bool, optional
            Whether the data should be independent and identically distributed between
            the clients or if the data should first be sorted by labels and distributed by chunks
            to each client (used to test the convergence in a worst case scenario), by default False    
        num_classes : int, optional
            The number of classes in the dataset, by default 10
        seed : int, optional
            Used to set a fix seed to replicate experiments, by default 42
        total_num_classes : int
            The number of classes in the dataset 
        lengths : List[int]
            A list with the length of each partition.
            """
    if iid or num_classes == total_num_classes:
        datasets = random_split(trainset, lengths, torch.Generator().manual_seed(seed))
        print("here")
    else:
        assert num_classes < total_num_classes, "num_classes must be less than or equal to total_num_classes"
        times = [0 for i in range(total_num_classes)]
        contain = []
        for i in range(num_clients):
            current = [i%total_num_classes]
            print("Current: ", current)
            times[i%total_num_classes] += 1
            print("Times: ", current)
            j = 1
            if i  == num_clients - 1:
                missing_labels = [i for i in range(total_num_classes) if times[i] == 0]
                print("missing_labels: ", missing_labels)
                if len(missing_labels) == num_classes:
                    current = missing_labels
                elif len(missing_labels) != 0:
                    for k in missing_labels:
                        current.append(k)
                        times[k] += 1
                    if len(missing_labels) != num_classes:
                        remaining_num_labels = num_classes - len(current)
                        # ind is a value between 0 and total_num_classes-1, excluding missing values
                        if remaining_num_labels > 0:
                            ind = random.sample([i for i in range(total_num_classes) if i not in missing_labels], remaining_num_labels)
                            for k in range(remaining_num_labels):
                                current.append(ind[k])
                                times[ind[k]] += 1
                else:
                    pass      
            else:       
                while (j < num_classes):             
                    ind = random.randint(0, total_num_classes-1)
                    print("Index: ", ind)
                    if (ind not in current):
                        j += 1
                        current.append(ind)
                        times[ind] += 1
                        print("times: ", times)
            contain.append(current)
            print("Client {} contains classes: {}".format(i, current))
        print("times: ", times)
        net_dataidx_map = {i:np.ndarray(0, dtype=np.int32) for i in range(num_clients)}
        for i in range(total_num_classes):
            idx_k = np.where(np.array(trainset.targets) == i)[0]
            print("Class {} has {} samples".format(i, len(idx_k)))
            np.random.shuffle(idx_k)
            split = np.array_split(idx_k, times[i])
            ids=0
            for j in range(num_clients):
                if i in contain[j]:
                    net_dataidx_map[j] = np.append(net_dataidx_map[j], split[ids])
                    ids += 1
        datasets = []
        for i in range(num_clients):
            datasets.append(Subset(trainset, net_dataidx_map[i]))

    print("Number of samples in each client:")
    for i in range(num_clients):
        print("Client {} has {} samples".format(i, len(datasets[i])))
    print("Number of samples in test set: {}".format(len(testset)))
    return datasets, testset