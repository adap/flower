import random
from collections import defaultdict

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST

def get_datasets(data_name, dataroot, normalize=True, val_size=10000):
    if data_name not in ["mnist", "cifar10", "cifar100"]:
        raise ValueError("Choose data_name from ['mnist', 'cifar10', 'cifar100']")

    normalization = {
        "mnist": transforms.Normalize((0.1307,), (0.3081,)),
        "cifar10": transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        "cifar100": transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    }[data_name]

    transform = transforms.Compose([transforms.ToTensor(), normalization])

    dataset = {
        "mnist": MNIST,
        "cifar10": CIFAR10,
        "cifar100": CIFAR100,
    }[data_name](dataroot, train=True, download=True, transform=transform)

    test_set = {
        "mnist": MNIST,
        "cifar10": CIFAR10,
        "cifar100": CIFAR100,
    }[data_name](dataroot, train=False, download=True, transform=transform)

    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    return train_set, val_set, test_set

def get_num_classes_samples(dataset):
    if isinstance(dataset, torch.utils.data.Subset):
        data_labels_list = np.array(dataset.dataset.targets)[list(dataset.indices)]
    else:
        data_labels_list = np.array(dataset.targets)

    classes, num_samples = np.unique(data_labels_list, return_counts=True)
    num_classes = len(classes)
    return num_classes, num_samples, data_labels_list


def gen_classes_per_node(dataset, num_users, classes_per_user=2, high_prob=0.6, low_prob=0.4):
    num_classes, num_samples, _ = get_num_classes_samples(dataset)
    count_per_class = (classes_per_user * num_users) // num_classes + 1
    class_dict = {}
    for i in range(num_classes):
        probs = np.random.uniform(low_prob, high_prob, size=count_per_class)
        probs_norm = (probs / probs.sum()).tolist()
        class_dict[i] = {"count": count_per_class, "prob": probs_norm}

    class_partitions = defaultdict(list)
    for i in range(num_users):
        c = []
        for _ in range(classes_per_user):
            class_counts = [class_dict[i]["count"] for i in range(num_classes)]
            max_class_counts = np.where(np.array(class_counts) == max(class_counts))[0]
            c.append(np.random.choice(max_class_counts))
            class_dict[c[-1]]["count"] -= 1
        class_partitions["class"].append(c)
        class_partitions["prob"].append([class_dict[i]["prob"].pop() for i in c])
    return class_partitions

def gen_data_split(dataset, num_users, class_partitions):
    num_classes, num_samples, data_labels_list = get_num_classes_samples(dataset)
    data_class_idx = {i: np.where(data_labels_list == i)[0] for i in range(num_classes)}

    for data_idx in data_class_idx.values():
        random.shuffle(data_idx)

    user_data_idx = [[] for _ in range(num_users)]
    for usr_i in range(num_users):
        for c, p in zip(class_partitions["class"][usr_i], class_partitions["prob"][usr_i]):
            end_idx = int(num_samples[c] * p)
            user_data_idx[usr_i].extend(data_class_idx[c][:end_idx])
            data_class_idx[c] = data_class_idx[c][end_idx:]

    return user_data_idx

def gen_random_loaders(data_name, data_path, num_users, bz, classes_per_user):
    loader_params = {
        "batch_size": bz,
        "shuffle": False,
        "pin_memory": True,
        "num_workers": 0,
    }
    dataloaders = []
    datasets = get_datasets(data_name, data_path, normalize=True)
    for i, d in enumerate(datasets):
        if i == 0:
            cls_partitions = gen_classes_per_node(d, num_users, classes_per_user)
            loader_params["shuffle"] = True
        usr_subset_idx = gen_data_split(d, num_users, cls_partitions)
        subsets = [torch.utils.data.Subset(d, x) for x in usr_subset_idx]
        dataloaders.append([torch.utils.data.DataLoader(x, **loader_params) for x in subsets])

    return dataloaders
