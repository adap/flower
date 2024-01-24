"""Handle the dataset partitioning and (optionally) complex downloads.

Please add here all the necessary logic to either download, uncompress, pre/post-process
your dataset (or all of the above). If the desired way of running your baseline is to
first download the dataset and partition it and then run the experiments, please
uncomment the lines below and tell us in the README.md (see the "Running the Experiment"
block) that this file should be executed first.
"""

import numpy as np
import torchvision.transforms as transforms

from moon.dataset import CIFAR10Sub, CIFAR100Sub


def load_cifar10_data(datadir):
    """Load CIFAR10 dataset."""
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10Sub(
        datadir, train=True, download=True, transform=transform
    )
    cifar10_test_ds = CIFAR10Sub(
        datadir, train=False, download=True, transform=transform
    )

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)


def load_cifar100_data(datadir):
    """Load CIFAR100 dataset."""
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100Sub(
        datadir, train=True, download=True, transform=transform
    )
    cifar100_test_ds = CIFAR100Sub(
        datadir, train=False, download=True, transform=transform
    )

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    return (X_train, y_train, X_test, y_test)


# pylint: disable=too-many-locals
def partition_data(dataset, datadir, partition, num_clients, beta):
    """Partition data into train and test sets for IID and non-IID experiments."""
    if dataset == "cifar10":
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == "cifar100":
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)

    n_train = y_train.shape[0]

    if partition in ("homo", "iid"):
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, num_clients)
        net_dataidx_map = {i: batch_idxs[i] for i in range(num_clients)}

    elif partition in ("noniid-labeldir", "noniid"):
        min_size = 0
        min_require_size = 10
        K = 10
        if dataset == "cifar100":
            K = 100
        elif dataset == "tinyimagenet":
            K = 200

        N = y_train.shape[0]
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, num_clients))
                proportions = np.array(
                    [
                        p * (len(idx_j) < N / num_clients)
                        for p, idx_j in zip(proportions, idx_batch)
                    ]
                )
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                ]
                min_size = min([len(idx_j) for idx_j in idx_batch])
        for j in range(num_clients):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    return (X_train, y_train, X_test, y_test, net_dataidx_map)
