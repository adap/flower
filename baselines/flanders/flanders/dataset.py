# Borrowed from adap/Flower examples

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import shutil
from PIL import Image
from torchvision.datasets import VisionDataset
from typing import Callable, Optional, Tuple, Any
from .dataset_preparation import create_lda_partitions
from sklearn.datasets import make_circles
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.utils import check_random_state

class Data(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len

def get_dataset(path_to_data: Path, cid: str, partition: str):

    # generate path to cid's data
    path_to_data = path_to_data / cid / (partition + ".pt")

    return TorchVision_FL(path_to_data, transform=cifar10Transformation())


def get_random_id_splits(total: int, val_ratio: float, shuffle: bool = True):
    """splits a list of length `total` into two following a
    (1-val_ratio):val_ratio partitioning.

    By default the indices are shuffled before creating the split and
    returning.
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


def do_fl_partitioning(path_to_dataset, pool_size, alpha, num_classes, val_ratio=0.0, seed=None):
    """Torchvision (e.g. CIFAR-10) datasets using LDA."""

    images, labels = torch.load(path_to_dataset)
    idx = np.array(range(len(images)))
    dataset = [idx, labels]
    partitions, _ = create_lda_partitions(
        dataset, num_partitions=pool_size, concentration=alpha, accept_imbalanced=True, seed=seed
    )

    # Show label distribution for first partition (purely informative)
    partition_zero = partitions[0][1]
    hist, _ = np.histogram(partition_zero, bins=list(range(num_classes + 1)))
    print(
        f"Class histogram for 0-th partition (alpha={alpha}, {num_classes} classes): {hist}"
    )

    # now save partitioned dataset to disk
    # first delete dir containing splits (if exists), then create it
    splits_dir = path_to_dataset.parent / "federated"
    if splits_dir.exists():
        shutil.rmtree(splits_dir)
    Path.mkdir(splits_dir, parents=True)

    for p in range(pool_size):

        labels = partitions[p][1]
        image_idx = partitions[p][0]
        imgs = images[image_idx]

        # create dir
        Path.mkdir(splits_dir / str(p))

        if val_ratio > 0.0:
            # split data according to val_ratio
            train_idx, val_idx = get_random_id_splits(len(labels), val_ratio)
            val_imgs = imgs[val_idx]
            val_labels = labels[val_idx]

            with open(splits_dir / str(p) / "val.pt", "wb") as f:
                torch.save([val_imgs, val_labels], f)

            # remaining images for training
            imgs = imgs[train_idx]
            labels = labels[train_idx]

        with open(splits_dir / str(p) / "train.pt", "wb") as f:
            torch.save([imgs, labels], f)

    return splits_dir


def cifar10Transformation():

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )


class TorchVision_FL(VisionDataset):
    """This is just a trimmed down version of torchvision.datasets.MNIST.

    Use this class by either passing a path to a torch file (.pt)
    containing (data, targets) or pass the data, targets directly
    instead.
    """

    def __init__(
        self,
        path_to_data=None,
        data=None,
        targets=None,
        transform: Optional[Callable] = None,
    ) -> None:
        path = path_to_data.parent if path_to_data else None
        super(TorchVision_FL, self).__init__(path, transform=transform)
        self.transform = transform

        if path_to_data:
            # load data and targets (path_to_data points to an specific .pt file)
            self.data, self.targets = torch.load(path_to_data)
        else:
            self.data = data
            self.targets = targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
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
        return len(self.data)


def get_cifar_10(path_to_data="datasets/cifar_nn/data"):
    """Downloads CIFAR10 dataset and generates a unified training set (it will
    be partitioned later using the LDA partitioning mechanism."""

    # download dataset and load train set
    train_set = datasets.CIFAR10(root=path_to_data, train=True, download=True)

    # fuse all data splits into a single "training.pt"
    data_loc = Path(path_to_data) / "cifar-10-batches-py"
    training_data = data_loc / "training.pt"
    print("Generating unified CIFAR dataset")
    torch.save([train_set.data, np.array(train_set.targets)], training_data)

    test_set = datasets.CIFAR10(
        root=path_to_data, train=False, transform=cifar10Transformation()
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
    """Helper function that loads both training and test datasets for MNIST.
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
        Total number of clients launched during training. This value dictates the number of unique to be created.
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

def get_circles(
    batch_size: int,
    n_samples: int = 1000,
    workers=1,
    is_train=True
):
    # Create a dataset with 10,000 samples.
    X, y = make_circles(n_samples = n_samples,
                        noise= 0.05,
                        random_state=26)
    if is_train:
        X_train, _, y_train, _ = train_test_split(X, y, test_size=.33)
        # Instantiate training and test data
        train_data = Data(X_train, y_train)
        dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=workers)
    else:
        _, X_test, _, y_test = train_test_split(X, y, test_size=.33)
        # Instantiate training and test data
        test_data = Data(X_test, y_test)
        dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=workers)
    return dataloader


def dataset_partitioner(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    client_id: int,
    number_of_clients: int,
    workers: int = 1,
) -> torch.utils.data.DataLoader:
    """Helper function to partition datasets
    Parameters
    ----------
    dataset: torch.utils.data.Dataset
        Dataset to be partitioned into *number_of_clients* subsets.
    batch_size: int
        Size of mini-batches used by the returned DataLoader.
    client_id: int
        Unique integer used for selecting a specific partition.
    number_of_clients: int
        Total number of clients launched during training. This value dictates the number of partitions to be created.
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
        dataset, batch_size=batch_size, shuffle=False, sampler=data_sampler, num_workers=workers
    )
    return data_loader

def get_partitioned_income(path: str, pool_size: int, train_size=0.8, test_size=0.2):
    data=pd.read_csv(path)
    copy=data
    encoder=OrdinalEncoder()
    encoded_values=encoder.fit_transform(data)
    data=pd.DataFrame(data=encoded_values, columns=copy.columns)
    Y=data["income"]
    data=data.loc[:,data.columns!="income"]

    X_train, X_test, y_train, y_test = [], [], [], []
    train_size = int((len(data) * train_size) // pool_size)+2
    test_size = int((len(data) * test_size) // pool_size)-2
    for i in range(pool_size):
        xtrain, xtest, ytrain, ytest = train_test_split(
            data, Y, 
            train_size=train_size, 
            test_size=test_size, 
            random_state=i, 
            shuffle=True,
            stratify=Y
        )
        X_train.append(xtrain)
        X_test.append(xtest)
        y_train.append(ytrain)
        y_test.append(ytest)

    return X_train, X_test, y_train, y_test