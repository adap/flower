"""Dataset utilities for FL experiments."""

# Borrowed from adap/Flower examples

import shutil
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from torchvision.datasets import VisionDataset

from .dataset_preparation import create_lda_partitions


class Data(torch.utils.data.Dataset):
    """Dataset class."""

    def __init__(self, X, y):
        """Initialize dataset."""
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        """Return data and label pair."""
        return self.X[index], self.y[index]

    def __len__(self):
        """Return size of dataset."""
        return self.len


def get_dataset(path_to_data: Path, cid: str, partition: str):
    """Return TorchVisionFL dataset object."""
    # generate path to cid's data
    path_to_data = path_to_data / cid / (partition + ".pt")

    return TorchVisionFL(path_to_data, transform=cifar10_transformation())


def get_dataloader(
    path_to_data: str, cid: str, is_train: bool, batch_size: int, workers: int
):
    """Generate trainset/valset object and returns appropiate dataloader."""
    partition = "train" if is_train else "val"
    dataset = get_dataset(Path(path_to_data), str(cid), partition)

    # we use as number of workers all the cpu cores assigned to this actor
    kwargs = {"num_workers": workers, "pin_memory": True, "drop_last": False}
    return DataLoader(dataset, batch_size=batch_size, **kwargs)


def get_random_id_splits(total: int, val_ratio: float, shuffle: bool = True):
    """Random split.

    Split a list of length `total` into two following a (1-val_ratio):val_ratio
    partitioning.

    By default the indices are shuffled before creating the split and returning.
    """
    if isinstance(total, int):
        indices = list(range(total))
    else:
        indices = total

    split = int(np.floor(val_ratio * len(indices)))
    # print(f"Users left out for validation (ratio={val_ratio}) = {split} ")
    if shuffle:
        np.random.shuffle(indices)
    return indices[split:], indices[:split]


# pylint: disable=too-many-arguments, too-many-locals
def do_fl_partitioning(
    path_to_dataset, pool_size, alpha, num_classes, val_ratio=0.0, seed=None
):
    """Torchvision (e.g. CIFAR-10) datasets using LDA."""
    images, labels = torch.load(path_to_dataset)
    idx = np.array(range(len(images)))
    dataset = [idx, labels]
    partitions, _ = create_lda_partitions(
        dataset,
        num_partitions=pool_size,
        concentration=alpha,
        accept_imbalanced=True,
        seed=seed,
    )

    # Show label distribution for first partition (purely informative)
    partition_zero = partitions[0][1]
    hist, _ = np.histogram(partition_zero, bins=list(range(num_classes + 1)))
    print(
        "Class histogram for 0-th partition"
        f"(alpha={alpha}, {num_classes} classes): {hist}"
    )

    # now save partitioned dataset to disk
    # first delete dir containing splits (if exists), then create it
    splits_dir = path_to_dataset.parent / "federated"
    if splits_dir.exists():
        shutil.rmtree(splits_dir)
    Path.mkdir(splits_dir, parents=True)

    for _, idx in enumerate(pool_size):
        labels = partitions[idx][1]
        image_idx = partitions[idx][0]
        imgs = images[image_idx]

        # create dir
        Path.mkdir(splits_dir / str(idx))

        if val_ratio > 0.0:
            # split data according to val_ratio
            train_idx, val_idx = get_random_id_splits(len(labels), val_ratio)
            val_imgs = imgs[val_idx]
            val_labels = labels[val_idx]

            with open(splits_dir / str(idx) / "val.pt", "wb") as fil:
                torch.save([val_imgs, val_labels], fil)

            # remaining images for training
            imgs = imgs[train_idx]
            labels = labels[train_idx]

        with open(splits_dir / str(idx) / "train.pt", "wb") as fil:
            torch.save([imgs, labels], fil)

    return splits_dir


def cifar10_transformation():
    """Return a torchvision.transforms object for CIFAR10 dataset."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )


class TorchVisionFL(VisionDataset):
    """TorchVision FL class.

    Use this class by either passing a path to a torch file (.pt) containing
    (data, targets) or pass the data, targets directly instead.

    This is just a trimmed down version of torchvision.datasets.MNIST.
    """

    def __init__(
        self,
        path_to_data=None,
        data=None,
        targets=None,
        transform: Optional[Callable] = None,
    ) -> None:
        """Initialize dataset."""
        path = path_to_data.parent if path_to_data else None
        super().__init__(path, transform=transform)
        self.transform = transform

        if path_to_data:
            # load data and targets (path_to_data points to an specific .pt file)
            self.data, self.targets = torch.load(path_to_data)
        else:
            self.data = data
            self.targets = targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Return a tuple (data, target)."""
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if not isinstance(img, Image.Image):  # if not PIL image
            if not isinstance(img, np.ndarray):  # if torch tensor
                img = img.numpy()

            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.data)


def get_cifar_10(path_to_data="flanders/datasets_files/cifar_10/data"):
    """Download CIFAR10 dataset."""
    # download dataset and load train set
    train_set = datasets.CIFAR10(root=path_to_data, train=True, download=True)

    # fuse all data splits into a single "training.pt"
    data_loc = Path(path_to_data) / "cifar-10-batches-py"
    training_data = data_loc / "training.pt"
    print("Generating unified CIFAR dataset")
    torch.save([train_set.data, np.array(train_set.targets)], training_data)

    test_set = datasets.CIFAR10(
        root=path_to_data, train=False, transform=cifar10_transformation()
    )

    # returns path where training data is and testset
    return training_data, test_set


def get_mnist(
    data_root: str,
    batch_size: int,
    cid: int,
    workers=1,
    nb_clients=10,
    is_train=True,
):
    """Load both training and test datasets for MNIST.

    Parameters
    ----------
    data_root: str
        Directory where MNIST dataset will be stored.
    train_batch_size: int
        Mini-batch size for training set.
    test_batch_size: int
        Mini-batch size for test set.
    cid: int
        Client ID used to select a specific partition.
    nb_clients: int
        Total number of clients launched during training.
        This value dictates the number of unique to be created.

    Returns
    -------
    (train_loader, test_loader): Tuple[DataLoader, DataLoader]
        Tuple contaning DataLoaders for training and test sets.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    if is_train:
        dataset = datasets.MNIST(
            data_root, train=True, download=True, transform=transform
        )
    else:
        dataset = datasets.MNIST(data_root, train=False, transform=transform)

    loader = dataset_partitioner(
        dataset=dataset,
        batch_size=batch_size,
        client_id=cid,
        number_of_clients=nb_clients,
        workers=workers,
    )

    return loader


def dataset_partitioner(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    client_id: int,
    number_of_clients: int,
    workers: int = 1,
) -> torch.utils.data.DataLoader:
    """Make datasets partitions for a specific client_id.

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
        Dataset to be partitioned into *number_of_clients* subsets.
    batch_size: int
        Size of mini-batches used by the returned DataLoader.
    client_id: int
        Unique integer used for selecting a specific partition.
    number_of_clients: int
        Total number of clients launched during training.
        This value dictates the number of partitions to be created.

    Returns
    -------
    data_loader: torch.utils.data.Dataset
        DataLoader for specific client_id considering number_of_clients partitions.
    """
    # Set the seed so we are sure to generate the same global batches
    # indices across all clients
    np.random.seed(123)

    # Get the data corresponding to this client
    dataset_size = len(dataset)
    nb_samples_per_clients = dataset_size // number_of_clients
    dataset_indices = list(range(dataset_size))
    np.random.shuffle(dataset_indices)

    # Get starting and ending indices w.r.t CLIENT_ID
    start_ind = int(client_id) * nb_samples_per_clients
    end_ind = start_ind + nb_samples_per_clients
    data_sampler = SubsetRandomSampler(dataset_indices[start_ind:end_ind])
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=data_sampler,
        num_workers=workers,
    )
    return data_loader


def get_partitioned_income(path: str, pool_size: int, train_size=0.8, test_size=0.2):
    """Return partitioned income dataset."""
    data = pd.read_csv(path)
    copy = data
    encoder = OrdinalEncoder()
    encoded_values = encoder.fit_transform(data)
    data = pd.DataFrame(data=encoded_values, columns=copy.columns)
    Y = data["income"]
    data = data.loc[:, data.columns != "income"]

    x_train, x_test, y_train, y_test = [], [], [], []
    train_size = int((len(data) * train_size) // pool_size) + 2
    test_size = int((len(data) * test_size) // pool_size) - 2
    for i in range(pool_size):
        xtrain, xtest, ytrain, ytest = train_test_split(
            data,
            Y,
            train_size=train_size,
            test_size=test_size,
            random_state=i,
            shuffle=True,
            stratify=Y,
        )
        x_train.append(xtrain)
        x_test.append(xtest)
        y_train.append(ytrain)
        y_test.append(ytest)

    return x_train, x_test, y_train, y_test


def get_partitioned_house(path: str, pool_size: int, train_size=0.8, test_size=0.2):
    """Return partitioned house dataset."""
    data = pd.read_csv(path)
    Y = data["median_house_value"].values
    data = data.drop(["median_house_value"], axis=1).values

    x_train, x_test, y_train, y_test = [], [], [], []
    train_size = int((len(data) * train_size) // pool_size) + 2
    test_size = int((len(data) * test_size) // pool_size) - 2
    for i in range(pool_size):
        xtrain, xtest, ytrain, ytest = train_test_split(
            data,
            Y,
            train_size=train_size,
            test_size=test_size,
            random_state=i,
            shuffle=True,
        )
        transf = preprocessing.RobustScaler()
        xtrain = transf.fit_transform(xtrain)
        xtest = transf.fit_transform(xtest)

        x_train.append(xtrain)
        x_test.append(xtest)
        y_train.append(ytrain)
        y_test.append(ytest)

    return x_train, x_test, y_train, y_test
