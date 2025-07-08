"""Dataset handling and partitioning for the Non-IID setting."""
import random
from collections import defaultdict

import numpy as np
import torch.utils.data
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST


def get_datasets(data_name, dataroot, normalize=True, val_size=10000):
    """Create train,val and test set splits for CIFAR10/100, MNIST datasets.

    Args:
        data_name: name of dataset, choose from [mnsit, cifar10, cifar100]
        dataroot: root to data dir
        normalize: True/False to normalize the data.
        val_size: validation split size (in #samples). By default, 10000.

    Returns
    -------
        tuple: train_set, val_set, test_set
    """
    if data_name == "mnist":
        normalization = transforms.Normalize((0.1307,), (0.3081,))
        data_obj = MNIST
    elif data_name == "cifar10":
        normalization = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        )
        data_obj = CIFAR10
    elif data_name == "cifar100":
        normalization = transforms.Normalize(
            (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        )
        data_obj = CIFAR100
    else:
        raise ValueError("choose data_name from ['mnist', 'cifar10', 'cifar100']")

    trans = [transforms.ToTensor()]

    if normalize:
        trans.append(normalization)

    transform = transforms.Compose(trans)

    dataset = data_obj(dataroot, train=True, download=True, transform=transform)

    test_set = data_obj(dataroot, train=False, download=True, transform=transform)

    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    return train_set, val_set, test_set


def get_num_classes_samples(dataset):
    """Extract information about certain dataset.

    Args:
        dataset: pytorch dataset object

    Returns
    -------
        tuple: number of classes, number of samples, list of labels.
    """
    # extract labels
    if isinstance(dataset, torch.utils.data.Subset):
        if isinstance(dataset.dataset.targets, list):
            data_labels_list = np.array(dataset.dataset.targets)[dataset.indices]
        else:
            data_labels_list = dataset.dataset.targets[dataset.indices]
    else:
        if isinstance(dataset.targets, list):
            data_labels_list = np.array(dataset.targets)
        else:
            data_labels_list = dataset.targets
    classes, num_samples = np.unique(data_labels_list, return_counts=True)
    num_classes = len(classes)
    return num_classes, num_samples, data_labels_list


def gen_classes_per_node(
    dataset, num_users, classes_per_user=2, high_prob=0.6, low_prob=0.4
):
    """Create the data distribution of each client.

    Args:
        dataset: pytorch dataset object
        num_users: number of clients
        - (Section `5.1. Heterogeneous Data` of paper)
        classes_per_user: number of classes assigned to each client
                          (2 for MNIST and CIFAR10, 10 for CIFAR100 )
        high_prob: highest prob sampled(0.6 as defined in section mentioned above)
        low_prob: lowest prob sampled(0.4 as defined in section mentioned above)

    Returns
    -------
        dict: mapping between classes and proportions
    """
    num_classes, _, _ = get_num_classes_samples(dataset)

    # divide classes and num samples for each user
    count_per_class = (classes_per_user * num_users) // num_classes + 1
    class_dict = {}
    for i in range(num_classes):
        # sampling alpha_i_c as given in the paper (section 5.1. Heterogeneous Data)
        probs = np.random.uniform(low_prob, high_prob, size=count_per_class)
        # normalizing
        probs_norm = (probs / probs.sum()).tolist()
        class_dict[i] = {"count": count_per_class, "prob": probs_norm}

    # assign each client with data indexes
    class_partitions = defaultdict(list)
    for i in range(num_users):
        c_max = []
        for _ in range(classes_per_user):
            class_counts = [class_dict[i]["count"] for i in range(num_classes)]
            max_class_counts = np.where(np.array(class_counts) == max(class_counts))[0]
            c_max.append(np.random.choice(max_class_counts))
            class_dict[c_max[-1]]["count"] -= 1
        class_partitions["class"].append(c_max)
        class_partitions["prob"].append([class_dict[i]["prob"].pop() for i in c_max])
    return class_partitions


def gen_data_split(dataset, num_users, class_partitions):
    """Divide data indexes for each client based on class_partition.

    Args:
        dataset: pytorch dataset object (train/val/test)
        num_users: number of clients
        class_partitions: proportion of classes per client

    Returns
    -------
        dict: mapping client to its indexes
    """
    num_classes, num_samples, data_labels_list = get_num_classes_samples(dataset)

    # create class index mapping
    data_class_idx = {i: np.where(data_labels_list == i)[0] for i in range(num_classes)}

    for data_idx in data_class_idx.values():
        random.shuffle(data_idx)

    # assigning samples to each client
    user_data_idx = [[] for i in range(num_users)]
    for usr_i in range(num_users):
        for class_usr, prob_usr in zip(
            class_partitions["class"][usr_i], class_partitions["prob"][usr_i]
        ):
            end_idx = int(num_samples[class_usr] * prob_usr)
            user_data_idx[usr_i].extend(data_class_idx[class_usr][:end_idx])
            data_class_idx[class_usr] = data_class_idx[class_usr][end_idx:]

    return user_data_idx


def gen_random_loaders(data_name, data_path, num_users, batch_size, classes_per_user):
    """Generate train,validation and test loaders for the clients.

    Args:
        data_name: name of dataset, choose from [mnist, cifar10, cifar100]
        data_path: root path for data dir
        num_users: number of clients
        batch_size: batch size
        - (Section `5.1. Heterogeneous Data` of paper)
        classes_per_user: number of classes assigned to each client
                          (2 in case of MNIST and CIFAR10, 10 in case of CIFAR100)

    Returns
    -------
        list: list of pytorch train, validation and test datalodes for all the clients
    """
    loader_params = {
        "batch_size": batch_size,
        "shuffle": False,
        "pin_memory": True,
        "num_workers": 0,
    }
    dataloaders = []
    datasets = get_datasets(data_name, data_path, normalize=True)
    for i, dataset in enumerate(datasets):
        # ensure same partition for train/test/val
        if i == 0:
            cls_partitions = gen_classes_per_node(dataset, num_users, classes_per_user)
            loader_params["shuffle"] = True
        usr_subset_idx = gen_data_split(dataset, num_users, cls_partitions)
        # create subsets for each client
        subsets = [torch.utils.data.Subset(dataset, x) for x in usr_subset_idx]
        # create dataloaders from subsets
        dataloaders.append(
            [torch.utils.data.DataLoader(x, **loader_params) for x in subsets]
        )

    return dataloaders
