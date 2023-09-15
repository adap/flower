"""Dataset loading and processing utilities."""

import pickle
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


def _get_dirichlet_data(y, num_clients, alpha, num_c, partition_equal=False):
    N = len(y)
    K = num_c
    n_nets = num_clients
    label_list_true = y
    net_dataidx_map = {}
    p_client = np.random.dirichlet(np.repeat(alpha, K), size=num_clients)

    idx_batch = [[] for _ in range(num_clients)]
    m = N // num_clients

    if not partition_equal:
        for k in range(K):
            idx_k = np.where(label_list_true == k)[0]
            np.random.shuffle(idx_k)
            proportions = p_client[:, k]
            proportions /= proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
            ]
    else:
        idx_labels = [np.where(label_list_true == k)[0] for k in range(K)]
        idx_counter = [0 for _ in range(K)]
        total_cnt = 0
        p_client_cdf = np.cumsum(p_client, axis=1)

        while total_cnt < m * num_clients:
            curr_clnt = np.random.randint(num_clients)
            if len(idx_batch[curr_clnt]) >= m:
                continue
            total_cnt += 1
            curr_prior = p_client_cdf[curr_clnt]
            while True:
                cls_label = np.argmax(np.random.uniform() <= curr_prior)
                if idx_counter[cls_label] >= len(idx_labels[cls_label]):
                    continue
                idx_batch[curr_clnt].append(
                    idx_labels[cls_label][idx_counter[cls_label]]
                )
                idx_counter[cls_label] += 1
                break

    for j in range(n_nets):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    net_cls_counts = {
        net_i: {u: c for u, c in zip(*np.unique(label_list_true[dataidx], return_counts=True))}
        for net_i, dataidx in net_dataidx_map.items()
    }

    local_sizes = np.array([len(net_dataidx_map[i]) for i in range(n_nets)])
    weights = local_sizes / np.sum(local_sizes)

    print("Data statistics: %s" % str(net_cls_counts))
    print("Data ratio: %s" % str(weights))

    return idx_batch


def _split_dataset(
    train_dataset, num_clients, alpha, num_classes, save_path, partition_equal=False
):
    print("Preparing dataset...")
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    x_train = next(iter(train_loader))[0].numpy()
    y_train = next(iter(train_loader))[1].numpy()
    indexes = _get_dirichlet_data(y_train, num_clients, alpha, num_classes, partition_equal=partition_equal)
    train_datasets = []
    for i, ind in enumerate(indexes):
        n_i = len(ind)
        x = x_train[ind]
        x_train = torch.Tensor(x[0:n_i])
        y = y_train[ind]
        y_train = torch.LongTensor(y[0:n_i])
        print(f"Client {i} : Training examples - {len(x_train)}")
        dataset_train_torch = TensorDataset(x_train, y_train)
        train_datasets.append(dataset_train_torch)

    with open(save_path, "wb") as file:
        pickle.dump(train_datasets, file)

    return train_datasets


def load_datasets(
    config, num_clients, batch_size, partition_equal=True
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
    trans_cifar = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]),
        ]
    )
    try:
        with open(ds_path, "rb") as file:
            train_datasets = pickle.load(file)
    except FileNotFoundError:
        dataset_train = Dataset(
            data_directory, train=True, download=True, transform=trans_cifar
        )
        train_datasets = _split_dataset(
            dataset_train,
            num_clients,
            config.alpha,
            config.num_classes,
            ds_path,
            partition_equal,
        )

    dataset_test = Dataset(
        data_directory, train=False, download=True, transform=trans_cifar
    )
    test_loader = DataLoader(dataset_test, batch_size=batch_size, num_workers=1)
    train_loaders = [
        DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=1)
        for ds in train_datasets
    ]

    return train_loaders, test_loader
