"""Handle the dataset partitioning and (optionally) complex downloads.

Please add here all the necessary logic to either download, uncompress, pre/post-process
your dataset (or all of the above). If the desired way of running your baseline is to
first download the dataset and partition it and then run the experiments, please
uncomment the lines below and tell us in the README.md (see the "Running the Experiment"
block) that this file should be executed first.
"""
import random
from collections import defaultdict

import numpy as np
from torch.utils.data import Dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class."""

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.targets = dataset.targets
        self.idxs = list(idxs)

    def __len__(self):
        """Return number of images."""
        return len(self.idxs)

    def __getitem__(self, item):
        """Return a transformed example of the dataset."""
        image, label = self.dataset[self.idxs[item]]
        return image, label


def iid(dataset, num_users):
    """Sample I.I.D. clients data from a dataset.

    Args:
        dataset: dataset object
        num_users: number of users
    Returns:
        dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, list(range(len(dataset)))
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


# pylint: disable=too-many-locals
def noniid(dataset, no_participants, alpha=0.5):
    """Sample non-I.I.D client data from dataset.

    Args:
        dataset: dataset object
        no_participants: number of users
        alpha: float parameter for dirichlet distribution
    Returns:
        dict of image index
    Requires:
        cifar_classes, a preprocessed class-indice dictionary.
        Sample Method: take a uniformly sampled 10/100-dimension vector
         as parameters for dirichlet distribution to sample number of
         images in each class.
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
            np.array(no_participants * [alpha])
        )
        for user in range(no_participants):
            no_imgs = int(round(sampled_probabilities[user]))
            datasize[user, n] = no_imgs
            sampled_list = cifar_classes[n][: min(len(cifar_classes[n]), no_imgs)]
            per_participant_list[user].extend(sampled_list)
            cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs) :]
    train_img_size = np.zeros(no_participants)
    for i in range(no_participants):
        train_img_size[i] = sum([datasize[i, j] for j in range(no_classes)])
    clas_weight = np.zeros((no_participants, no_classes))
    for i in range(no_participants):
        for j in range(no_classes):
            clas_weight[i, j] = float(datasize[i, j]) / float((train_img_size[i]))
    return per_participant_list, clas_weight
