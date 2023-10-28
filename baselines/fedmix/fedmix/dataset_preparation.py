"""..."""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Subset, TensorDataset
import os
import random
import json
from torchvision.datasets import CIFAR10, CIFAR100


def _download_cifar10():
    """..."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = CIFAR10("./dataset", train=True, download=True, transform=transform)
    testset = CIFAR10("./dataset", train=False, download=True, transform=transform)
    return trainset, testset


def _download_cifar100():
    """..."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    trainset = CIFAR100("./dataset", train=True, download=True, transform=transform)
    testset = CIFAR100("./dataset", train=False, download=True, transform=transform)
    return trainset, testset


def _partition_cifar(
    trainset, num_classes, num_clients, num_classes_per_client, seed
):
    """..."""
    partition_size = int(len(trainset) / num_clients)
    np.random.seed(seed)

    class_indices = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(trainset):
        class_indices[label].append(idx)

    client_datasets = []

    for _client_id in range(num_clients):
        selected_classes = np.random.choice(
            num_classes, num_classes_per_client, replace=False
        )
        selected_indices = []

        for class_idx in selected_classes:
            selected_indices.extend(class_indices[class_idx])

        np.random.shuffle(selected_indices)
        client_indices = selected_indices[:partition_size]

        client_dataset = Subset(trainset, client_indices)
        client_datasets.append(client_dataset)

    return client_datasets


def _partition_cifar_new(
    trainset, num_classes, num_clients, num_classes_per_client, seed
):
    """..."""
    labels = set(range(num_classes))
    np.random.seed(seed)

    nlbl = [
        np.random.choice(len(labels), num_classes_per_client, replace=False)
        for u in range(num_clients)
    ]
    check = set().union(*[set(a) for a in nlbl])

    while len(check) < len(labels):
        missing = labels - check
        for m in missing:
            nlbl[np.random.randint(0, num_clients)][
                np.random.randint(0, num_classes_per_client)
            ] = m
        check = set().union(*[set(a) for a in nlbl])

    class_map = {c: [u for u, lbl in enumerate(nlbl) if c in lbl] for c in labels}
    assignment = np.zeros(len(trainset))
    targets = np.array(trainset.targets)

    for lbl, users in class_map.items():
        ids = np.where(targets == lbl)[0]
        assignment[ids] = np.random.choice(users, len(ids))

    dataset_indices = [np.where(assignment == i)[0] for i in range(num_clients)]

    return [Subset(trainset, ind) for ind in dataset_indices]


def _download_femnist(num_clients):
    os.system(f'cd fedmix/femnist && ./preprocess.sh -s niid --iu {num_clients / 3550} --sf 0.1 -t sample')


def _partition_femnist(num_clients):
    train_path = 'fedmix/femnist/data/train'
    train_json_files = [f for f in os.listdir(train_path) if f.endswith('.json')]
    client_datasets_dict = {}

    train_user_ids = []

    for train_json in train_json_files:
        with open(os.path.join(train_path, train_json), 'r') as file:
            data = json.load(file)
            user_data = data['user_data']
            for user_id, user_info in user_data.items():
                train_user_ids.append(user_id)
                x = np.array(user_info['x'])
                x = x.reshape(-1, 1, 28, 28)
                y = np.array(user_info['y'])

                x = torch.tensor(x, dtype=torch.float32)
                y = torch.tensor(y)

                client_datasets_dict[user_id] = TensorDataset(x, y)

    user_ids = list(client_datasets_dict.keys())
    selected_user_ids = random.sample(user_ids, num_clients)

    print('selected users:', selected_user_ids)

    client_datasets = []
    for user_id in selected_user_ids:
        client_datasets.append(client_datasets_dict[user_id])

    test_path = 'fedmix/femnist/data/test'
    test_json_files = [f for f in os.listdir(test_path) if f.endswith('.json')]
    test_x, test_y = [], []

    test_user_ids = []

    for test_json in test_json_files:
        with open(os.path.join(test_path, test_json), 'r') as file:
            data = json.load(file)
            user_data = data['user_data']
            for user_id, user_info in user_data.items():
                test_user_ids.append(user_id)
                if user_id in selected_user_ids:
                    x = user_info['x']
                    y = user_info['y']

                    test_x.extend(x)
                    test_y.extend(y)

    x = torch.tensor(np.array(test_x).reshape(-1, 1, 28, 28), dtype=torch.float32)
    y = torch.tensor(np.array(test_y))
    testset = TensorDataset(x, y)

    return client_datasets, testset


def _mash_data(client_datasets, mash_batch_size, num_classes):
    """..."""
    mashed_data = []
    for client_dataset in client_datasets:
        mashed_image, mashed_label = [], []

        for i, (image, label) in enumerate(client_dataset):
            mashed_image.append(image)
            mashed_label.append(torch.tensor([label]))

            if (not mash_batch_size == "all") and (i + 1) % mash_batch_size == 0:
                mashed_data.append(
                    (
                        torch.squeeze(
                            torch.mean(
                                torch.stack(mashed_image[-mash_batch_size:]), dim=0
                            )
                        ),
                        torch.mean(
                            F.one_hot(
                                torch.squeeze(
                                    torch.stack(mashed_label[-mash_batch_size:])
                                ),
                                num_classes,
                            ).to(dtype=torch.float32),
                            dim=0,
                        ),
                    )
                )

        if mash_batch_size == "all":
            mashed_data.append(
                (
                    torch.squeeze(torch.mean(torch.stack(mashed_image), dim=0)),
                    torch.mean(
                        F.one_hot(
                            torch.squeeze(torch.stack(mashed_label)), num_classes
                        ).to(dtype=torch.float32),
                        dim=0,
                    ),
                )
            )

        mashed_image, mashed_label = [], []

    return mashed_data
