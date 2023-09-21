from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, random_split
from torchvision.datasets import CIFAR10

from config import PARAMS


def load_datasets(iid=True):
    if iid:
        return load_iid_data()
    else:
        train_dataset_cifar, user_groups_train_cifar = get_dataset_cifar10_noniid(
            PARAMS.num_clients,
            PARAMS.nclass_cifar,
            PARAMS.nsamples_cifar,
            PARAMS.rate_unbalance_cifar,
        )
        return load_noniid_data(train_dataset_cifar, user_groups_train_cifar)


def load_iid_data():
    train_batch_size = 4

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = CIFAR10("./dataset", train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size, shuffle=True
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    # functions to show an image
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.figure(figsize=(5, 2))
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    # get some random training images
    dataiter = iter(dataloader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(" ".join(f"{classes[labels[j]]:5s}" for j in range(train_batch_size)))

    # ----------------------------------

    # Split training set into N partitions to simulate the individual dataset
    partition_size = len(dataset) // PARAMS.num_clients
    lengths = [partition_size] * PARAMS.num_clients
    datasets = random_split(dataset, lengths)
    trainloaders = []

    for ds in datasets:
        trainloaders.append(
            torch.utils.data.DataLoader(ds, batch_size=train_batch_size, shuffle=True)
        )

    return trainloaders


def get_dataset_cifar10_noniid(num_users, n_class, nsamples, rate_unbalance):
    apply_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_dataset = CIFAR10(
        "./dataset", train=True, download=True, transform=apply_transform
    )
    # Chose equal splits for every user
    user_groups_train = cifar_extr_noniid(
        train_dataset, num_users, n_class, nsamples, rate_unbalance
    )
    return train_dataset, user_groups_train


def cifar_extr_noniid(train_dataset, num_users, n_class, num_samples, rate_unbalance):
    num_shards_train, num_imgs_train = int(50000 / num_samples), num_samples
    num_classes = 10
    print(num_shards_train, num_imgs_train)
    assert n_class * num_users <= num_shards_train
    assert n_class <= num_classes
    idx_shard = [i for i in range(num_shards_train)]
    dict_users_train = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards_train * num_imgs_train)
    labels = np.array(train_dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]

    # divide and assign
    for i in range(num_users):
        user_labels = np.array([])
        rand_set = set(np.random.choice(idx_shard, n_class, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        unbalance_flag = 0
        for rand in rand_set:
            if unbalance_flag == 0:
                dict_users_train[i] = np.concatenate(
                    (
                        dict_users_train[i],
                        idxs[rand * num_imgs_train : (rand + 1) * num_imgs_train],
                    ),
                    axis=0,
                )
                user_labels = np.concatenate(
                    (
                        user_labels,
                        labels[rand * num_imgs_train : (rand + 1) * num_imgs_train],
                    ),
                    axis=0,
                )
            else:
                dict_users_train[i] = np.concatenate(
                    (
                        dict_users_train[i],
                        idxs[
                            rand
                            * num_imgs_train : int(
                                (rand + rate_unbalance) * num_imgs_train
                            )
                        ],
                    ),
                    axis=0,
                )
                user_labels = np.concatenate(
                    (
                        user_labels,
                        labels[
                            rand
                            * num_imgs_train : int(
                                (rand + rate_unbalance) * num_imgs_train
                            )
                        ],
                    ),
                    axis=0,
                )
            unbalance_flag = 1

    return dict_users_train


def load_noniid_data(train_dataset_cifar, user_groups_train_cifar):
    for client_no, array in user_groups_train_cifar.items():
        class_no = []

        for idx in array:
            class_no.append(train_dataset_cifar[int(idx)][1])
        counts = Counter(class_no)
        print(client_no, counts)

    # combine all index list into one nested list
    indices = [val for d in [user_groups_train_cifar] for val in d.values()]
    indices = [list(a) for a in indices]
    indices = [[int(val) for val in sublist] for sublist in indices]

    trainloaders = []

    for index_list in indices:
        subset = Subset(train_dataset_cifar, index_list)
        trainloaders.append(
            torch.utils.data.DataLoader(subset, batch_size=4, shuffle=True)
        )

    return trainloaders
