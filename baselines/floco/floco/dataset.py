"""floco: A Flower Baseline."""

import os
from typing import Tuple

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from flwr.common import Context


def load_dataloaders(
    partition_id: int, num_partitions: int, context: Context
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create dataloaders for a specified dataset and partition.

    partition_id : int
        The ID of the partition to load.
    num_partitions : int
        The total number of partitions.
    dataset : str, optional
        The name of the dataset to load, by default "CIFAR10".
    dataset_split : str, optional
        The method to split the dataset, by default "Dirichlet".
    dataset_split_arg : float, optional
        The argument for the dataset split method, by default 0.5.
        The seed for random number generation, by default 0.
    batch_size : int, optional
        The batch size for the dataloaders, by default 50.

    Tuple[DataLoader, DataLoader, DataLoader]
        A tuple containing the training DataLoader, validation DataLoader,
        and test DataLoader for the specified partition.
    """
    dataset = str(context.run_config["dataset"])
    dataset_split = str(context.run_config["dataset-split"])
    dataset_split_arg = float(context.run_config["dataset-split-arg"])
    seed = int(context.run_config["dataset-split-arg"])
    batch_size = int(context.run_config["batch-size"])

    trainloaders, valloaders, testloader = load_image_dataset(
        dataset,
        seed,
        dataset_split,
        batch_size=batch_size,
        unbalanced_sgm=0,
        rule_arg=dataset_split_arg,
        data_path="data/",
        val_frac=0.2,
        num_clients=num_partitions,
    )
    return trainloaders[partition_id], valloaders[partition_id], testloader


def load_image_dataset(
    dataset,
    seed,
    rule,
    batch_size=50,
    unbalanced_sgm=0,
    rule_arg="",
    data_path=".",
    val_frac=0.2,
    num_clients=100,
):
    """Load image dataset for federated learning."""
    name = f"{dataset}_{num_clients}_{seed}_{rule}_{rule_arg}"
    if unbalanced_sgm != 0:
        name += f"_{unbalanced_sgm}"

    # Transforms
    if dataset == "CIFAR10":
        img_transforms = None
        normalization = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201]
        )
    else:
        raise NotImplementedError("Dataset not implemented")

    # 0. Prepare data-directories if not ready
    path = f"{data_path}/{name}"
    path_exists = os.path.exists(path)
    if path_exists:
        client_x, client_y, test_x, test_y = _load_preprocessed_data(
            f"{data_path}/{name}"
        )
        num_classes = len(np.unique(test_y))
        if rule == "Fold":
            folds = _compute_folds(num_classes, indices_or_sections=rule_arg)
        else:
            folds = None
    else:
        # 1. Get Raw data
        if dataset == "CIFAR10":
            ds = getattr(torchvision.datasets, dataset)
            trainset = ds(
                root=f"{data_path}/Raw", train=True, download=True, transform=None
            )
            testset = ds(
                root=f"{data_path}/Raw", train=False, download=True, transform=None
            )
            train_x = trainset.data
            train_y = np.array(trainset.targets)
            test_x = testset.data
            test_y = np.array(testset.targets)
        else:
            raise NotImplementedError(f"Dataset '{dataset}' not implemented")

        num_classes = len(np.unique(train_y))

        # 2. Shuffle Data
        np.random.seed(seed)
        train_rand_perm = np.random.permutation(len(train_y))
        train_x = train_x[train_rand_perm]
        train_y = train_y[train_rand_perm]
        test_rand_perm = np.random.permutation(len(test_y))
        test_x = test_x[test_rand_perm]
        test_y = test_y[test_rand_perm]

        # 3. Create list fo each client dataset

        folds = None
        if rule == "iid":
            client_data_list = _client_data_list(train_y, num_clients, unbalanced_sgm)
            cum_sum_list = np.concatenate(([0], np.cumsum(client_data_list)))
            client_x = [
                np.array(train_x[cum_sum_list[i] : cum_sum_list[i + 1]])
                for i in range(num_clients)
            ]
            client_y = [
                np.array(train_y[cum_sum_list[i] : cum_sum_list[i + 1]])
                for i in range(num_clients)
            ]
        elif rule == "Dirichlet":
            client_x, client_y = dirichlet_split(
                train_x,
                train_y,
                num_classes,
                num_clients,
                alpha=rule_arg,
                unbalanced_sgm=unbalanced_sgm,
            )
        elif rule == "Fold":
            folds = _compute_folds(num_classes, indices_or_sections=rule_arg)
            client_x, client_y = fold_split(train_x, train_y, folds, num_clients)
        else:
            raise NotImplementedError(f"Unknown rule '{rule}'.")

        # Save data
        _save_preprocessed_data(
            client_x, client_y, test_x, test_y, path=f"{data_path}/{name}"
        )

    # Create dataloaders
    trainloaders = []
    valloaders = []
    for i in range(num_clients):
        train_x, val_x = random_split(
            client_x[i],
            [1 - val_frac, val_frac],
            generator=torch.Generator().manual_seed(i),
        )
        train_y, val_y = random_split(
            client_y[i],
            [1 - val_frac, val_frac],
            generator=torch.Generator().manual_seed(i),
        )

        trainloaders.append(
            DataLoader(
                ImageDataset(
                    data_x=train_x,
                    data_y=train_y,
                    img_transforms=img_transforms,
                    normalization=normalization,
                    train=True,
                ),
                batch_size=batch_size,
                shuffle=True,
            )
        )
        valloaders.append(
            DataLoader(
                ImageDataset(
                    data_x=val_x,
                    data_y=val_y,
                    normalization=normalization,
                    train=False,
                ),
                batch_size=batch_size,
                shuffle=False,
            )
        )
    testloader = DataLoader(
        ImageDataset(
            data_x=test_x,
            data_y=test_y,
            normalization=normalization,
            train=False,
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    return trainloaders, valloaders, testloader


def dirichlet_split(train_x, train_y, num_classes, num_clients, alpha, unbalanced_sgm):
    """Generate a Dirichlet data split."""
    client_x = [[] for _ in range(num_clients)]
    client_y = [[] for _ in range(num_clients)]

    # Class-dirichlet split. Unequal ds size splitting technique from:
    # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
    if alpha == 0.1:
        client_to_data_ids = {k: [] for k in range(num_clients)}
        for label_id in range(len(np.unique(train_y))):
            idx_batch = [[] for _ in range(num_clients)]
            label_ids = np.where(train_y == label_id)[0]
            label_proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            label_proportions = np.cumsum(label_proportions * len(label_ids)).astype(
                int
            )[:-1]
            idx_batch = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(
                    idx_batch, np.array_split(label_ids, label_proportions)
                )
            ]
            for client_id in range(num_clients):
                client_to_data_ids[client_id] += idx_batch[client_id]

        if isinstance(train_x, np.ndarray):
            for k, v in client_to_data_ids.items():
                client_x[k] = train_x[v]
                client_y[k] = train_y[v]
        else:
            for k, v in client_to_data_ids.items():
                client_x[k] = [train_x[i] for i in v]
                client_y[k] = [train_y[i] for i in v]

    # Client-dirichlet split. Equal ds size splitting technique from:
    # https://github.com/gaoliang13/FedDC/blob/main/utils_dataset.py
    elif alpha in [0.3, 0.5]:
        client_data_list = _client_data_list(train_y, num_clients, unbalanced_sgm)
        class_priors = np.random.dirichlet(
            alpha=[alpha] * num_classes, size=num_clients
        )
        prior_cumsum = np.cumsum(class_priors, axis=1)
        idx_list = [np.where(train_y == i)[0] for i in range(num_classes)]
        class_amount = [len(idx_list[i]) for i in range(num_classes)]
        while np.sum(client_data_list) != 0:
            i = np.random.randint(num_clients)
            # If current node is full resample a client
            # ##print('Remaining Data: %d' %np.sum(client_data_list))
            if client_data_list[i] <= 0:
                continue
            client_data_list[i] -= 1
            curr_prior = prior_cumsum[i]
            while True:
                class_label = np.argmax(np.random.uniform() <= curr_prior)
                # Redraw class label if train_y is out of that class
                if class_amount[class_label] <= 0:
                    continue
                class_amount[class_label] -= 1
                client_x[i].append(
                    train_x[idx_list[class_label][class_amount[class_label]]]
                )
                client_y[i].append(
                    train_y[idx_list[class_label][class_amount[class_label]]]
                )
                break
        class_means = np.zeros((num_clients, num_classes))
        for client in range(num_clients):
            for class_ in range(num_classes):
                class_means[client, class_] = np.mean(client_y[client] == class_)

    client_x = [np.array(x) for x in client_x]
    client_y = [np.array(y) for y in client_y]
    return client_x, client_y


def fold_split(train_x, train_y, folds, num_clients):
    """Generate a fold data split as introduced in https://arxiv.org/abs/2007.03797."""
    client_x = [[] for _ in range(num_clients)]
    client_y = [[] for _ in range(num_clients)]

    dict_users = _get_fold_data_ids(
        train_y=train_y, num_clients=num_clients, folds=folds
    )
    for i, data_ids in enumerate(dict_users.values()):
        client_x[i].append(train_x[data_ids])
        client_y[i].append(train_y[data_ids])

    client_x = [np.concatenate(x) for x in client_x]
    client_y = [np.concatenate(y) for y in client_y]
    return client_x, client_y


def _compute_folds(num_classes, indices_or_sections):
    return np.array_split(
        ary=np.arange(num_classes), indices_or_sections=indices_or_sections
    )


def _save_preprocessed_data(client_x: list, client_y: list, test_x, test_y, path: str):
    os.makedirs(f"{path}")
    for i, (clx, cly) in enumerate(zip(client_x, client_y)):
        np.save(f"{path}/client_x_{i}.npy", clx, allow_pickle=True)
        np.save(f"{path}/client_y_{i}.npy", cly, allow_pickle=True)
    np.save(f"{path}/test_x.npy", test_x, allow_pickle=True)
    np.save(f"{path}/test_y.npy", test_y, allow_pickle=True)


def _load_preprocessed_data(path: str):
    num_clients = len([f for f in os.listdir(path) if f.startswith("client_x_")])
    client_x = [
        np.load(f"{path}/client_x_{i}.npy", allow_pickle=True)
        for i in range(num_clients)
    ]
    client_y = [
        np.load(f"{path}/client_y_{i}.npy", allow_pickle=True)
        for i in range(num_clients)
    ]
    test_x = np.load(f"{path}/test_x.npy", allow_pickle=True)
    test_y = np.load(f"{path}/test_y.npy", allow_pickle=True)
    return client_x, client_y, test_x, test_y


class ImageDataset(Dataset):
    """A custom dataset class for handling image data.

    Args:
        data_x (numpy.ndarray): Array of image data.
        data_y (list or numpy.ndarray): Array of labels corresponding to the image data.
        img_transforms (callable, optional): A function/transform to apply to the images
            during training. Default is None.
        normalization (callable): A function/transform to normalize the images.
        train (bool, optional): Flag indicating whether the dataset is used for
        training. Default is False.

    Attributes
    ----------
        X_data (numpy.ndarray): Array of image data.
        Y_data (torch.LongTensor): Tensor of labels corresponding to the image data.
        pil_transform (callable): Transform to convert numpy array to PIL image.
        img_transforms (callable): A function/transform to apply to the images
            during training.
        tensor_transform (callable): Transform to convert PIL image to tensor.
        normalization (callable): A function/transform to normalize the images.
        train (bool): Flag indicating whether the dataset is used for training.

    Methods
    -------
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns the image and label at the specified index.
    """

    def __init__(
        self, data_x, data_y, img_transforms=None, normalization=None, train=False
    ):
        self.x_data = data_x
        self.y_data = torch.LongTensor(data_y)
        self.pil_transform = transforms.ToPILImage()
        self.img_transforms = img_transforms
        self.tensor_transform = transforms.ToTensor()
        self.normalization = normalization
        self.train = train

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.x_data)

    def __getitem__(self, idx):
        """Get element at index idx in the dataset."""
        x = np.array(self.x_data[idx], copy=True)
        if self.train:
            if self.img_transforms is not None:
                x = self.pil_transform(x)
                x = self.img_transforms(x)
        x = self.tensor_transform(x)
        x = self.normalization(x)
        y = self.y_data[idx]
        return x, y


def _client_data_list(train_y, num_clients, unbalanced_sgm):
    """Define number of data points per client (ONLY IID AND DIRICHLET)."""
    n_data_per_client = int((len(train_y)) / num_clients)
    # Draw from lognormal distribution
    client_data_list = np.random.lognormal(
        mean=np.log(n_data_per_client), sigma=unbalanced_sgm, size=num_clients
    )
    client_data_list = (
        client_data_list / np.sum(client_data_list) * len(train_y)
    ).astype(int)

    # Add/Subtract the excess number starting from first client
    diff = np.sum(client_data_list) - len(train_y)
    if diff != 0:
        for client_i in range(num_clients):
            if client_data_list[client_i] > diff:
                client_data_list[client_i] -= diff
                break

    return client_data_list


def _get_fold_data_ids(
    train_y, num_clients, folds, seed=0, noniid_s=20, local_size=500, train=True
):
    np.random.seed(seed)
    s = noniid_s / 100
    num_per_user = local_size if train else 300
    num_classes = len(np.unique(train_y))
    # -------------------------------------------------------
    # divide the first dataset that all clients share (includes all classes)
    num_imgs_iid = int(num_per_user * s)
    num_imgs_noniid = num_per_user - num_imgs_iid
    dict_users = {i: np.array([]) for i in range(num_clients)}
    num_samples = len(train_y)
    num_per_label_total = int(num_samples / num_classes)
    labels1 = np.array(train_y)
    idxs1 = np.arange(len(train_y))
    # iid labels
    idxs_labels = np.vstack((idxs1, labels1))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # label available
    label_list = list(range(num_classes))
    # number of imgs has allocated per label
    label_used = (
        [2000 for i in range(num_classes)]
        if train
        else [500 for i in range(num_classes)]
    )
    iid_per_label = int(num_imgs_iid / num_classes)
    iid_per_label_last = num_imgs_iid - (num_classes - 1) * iid_per_label

    np.random.seed(seed)
    for i in range(num_clients):
        # allocate iid idxs
        label_cnt = 0
        for y in label_list:
            label_cnt = label_cnt + 1
            iid_num = iid_per_label
            start = y * num_per_label_total + label_used[y]
            if label_cnt == num_classes:
                iid_num = iid_per_label_last
            if (label_used[y] + iid_num) > num_per_label_total:
                start = y * num_per_label_total
                label_used[y] = 0
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[start : start + iid_num]), axis=0
            )
            label_used[y] = label_used[y] + iid_num
        # allocate noniid idxs
        # rand_label = np.random.choice(label_list, 3, replace=False)
        rand_label = folds[i % len(folds)]
        noniid_labels = len(rand_label)
        noniid_per_num = int(num_imgs_noniid / noniid_labels)
        noniid_per_num_last = num_imgs_noniid - noniid_per_num * (noniid_labels - 1)
        label_cnt = 0
        for y in rand_label:
            label_cnt = label_cnt + 1
            noniid_num = noniid_per_num
            start = y * num_per_label_total + label_used[y]
            if label_cnt == noniid_labels:
                noniid_num = noniid_per_num_last
            if (label_used[y] + noniid_num) > num_per_label_total:
                start = y * num_per_label_total
                label_used[y] = 0
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[start : start + noniid_num]), axis=0
            )
            label_used[y] = label_used[y] + noniid_num
        dict_users[i] = dict_users[i].astype(int)

    return dict_users
