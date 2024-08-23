"""Handle the dataset partitioning and (optionally) complex downloads.

Please add here all the necessary logic to either download, uncompress, pre/post-process
your dataset (or all of the above). If the desired way of running your baseline is to
first download the dataset and partition it and then run the experiments, please
uncomment the lines below and tell us in the README.md (see the "Running the Experiment"
block) that this file should be executed first.
"""

import os
import random
import shutil
import tarfile
import urllib

from logging import DEBUG, INFO
from flwr.common.logger import log

import torch
import torchvision


from PIL import Image
from torch.utils.model_zoo import tqdm
from torchvision.datasets import MNIST
from torchvision.datasets.utils import check_integrity


def _gen_bar_updater():
    pbar = tqdm(total=None)
    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)
    return bar_update


def download_url(url: str, root: str, filename=None, md5=None) -> None:
    """Download a file from a url and place it in root.

    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under.
        If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    # check if file is already present locally
    if check_integrity(fpath, md5):
        print("Using downloaded and verified file: " + fpath)
    else:  # download the file
        try:
            print("Downloading " + url + " to " + fpath)
            urllib.request.urlretrieve(url, fpath, reporthook=_gen_bar_updater())
        except (urllib.error.URLError, IOError) as temp_except:
            if url[:5] == "https":
                url = url.replace("https:", "http:")
                print(
                    "Failed download. Trying https -> http instead."
                    " Downloading " + url + " to " + fpath
                )
                urllib.request.urlretrieve(url, fpath, reporthook=_gen_bar_updater())
            else:
                raise temp_except
        # check integrity of downloaded file
        if not check_integrity(fpath, md5):
            raise RuntimeError("File not found or faultyed.")


def _is_tarxz(filename: str) -> bool:
    return filename.endswith(".tar.xz")


def _is_tar(filename: str) -> bool:
    return filename.endswith(".tar")


def _is_targz(filename: str) -> bool:
    return filename.endswith(".tar.gz")


def _is_tgz(filename: str) -> bool:
    return filename.endswith(".tgz")


def _extract_archive(
    from_path: str, to_path=None, remove_finished: bool = False
) -> None:
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_tar(from_path):
        with tarfile.open(from_path, "r") as tar:
            tar.extractall(path=to_path)
    elif _is_targz(from_path) or _is_tgz(from_path):
        with tarfile.open(from_path, "r:gz") as tar:
            tar.extractall(path=to_path)
    elif _is_tarxz(from_path):
        with tarfile.open(from_path, "r:xz") as tar:
            tar.extractall(path=to_path)
    else:
        raise ValueError("Extraction of {} not supported".format(from_path))

    if remove_finished:
        os.remove(from_path)


def _download_and__extract_archive(
    url: str,
    download_root: str,
    extract_root=None,
    filename=None,
    md5=None,
    remove_finished: bool = False,
) -> None:
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print("Extracting {} to {}".format(archive, extract_root))
    _extract_archive(archive, extract_root, remove_finished)


class FEMNIST(MNIST):
    """The dataset is derived from the Leaf repository.

    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST dataset,
    grouping examples by writer. Details about Leaf were published in "LEAF: A Benchmark
    for Federated Settings"
    https://arxiv.org/abs/1812.01097.
    """

    resources = [
        (
            "https://raw.githubusercontent.com/tao-shen/FEMNIST_pytorch/master/femnist.tar.gz",
            "59c65cec646fc57fe92d27d83afdf0ed",
        )
    ]

    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=True
    ) -> None:

        super(MNIST, self).__init__(root, transform=transform)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if self._check_legacy_exist():
            self.data, self.targets, self.clients = self._load_legacy_data()
            return

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        self.data, self.targets, self.clients = self._load_data()

    def _check_exists(self) -> bool:
        return os.path.isfile(
            os.path.join(
                self.processed_folder,
                self.training_file if self.train else self.test_file,
            )
        )

    def _load_data(self):
        data_file = self.training_file if self.train else self.test_file
        data, targets, clients = torch.load(
            os.path.join(self.processed_folder, data_file)
        )
        return data, targets, clients

    def __getitem__(self, index: int):
        """Return the item at the given index."""
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode="F")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def download(self) -> None:
        """Download the FEMNIST data if it doesn't exist in processed_folder already."""
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition("/")[2]
            _download_and__extract_archive(
                url, download_root=self.raw_folder, filename=filename, md5=md5
            )

        fpath_train = os.path.join(self.raw_folder, self.training_file)
        fpath_test = os.path.join(self.raw_folder, self.test_file)
        shutil.move(fpath_train, self.processed_folder)
        shutil.move(fpath_test, self.processed_folder)

    def _load_legacy_data(self):
        data_file = self.training_file if self.train else self.test_file
        data, targets, clients = torch.load(
            os.path.join(self.processed_folder, data_file)
        )
        return data, targets, clients

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.data)


def _get_train_val_datasets(dataset_name, data_dir):
    resize_trasnform = torchvision.transforms.Resize((32, 32))
    if dataset_name == "cifar10":
        train_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        test_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=train_transforms
        )

        val_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=test_transforms
        )

        return train_dataset, val_dataset, 10

    elif dataset_name == "femnist":
        transform = torchvision.transforms.Compose(
            [
                resize_trasnform,
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        train_ds = FEMNIST(data_dir, train=True, transform=transform, download=True)
        test_ds = FEMNIST(data_dir, train=False, transform=transform, download=True)
        train_ds = SubsetToDataset(train_ds)
        valid_ds = SubsetToDataset(test_ds)

        return train_ds, valid_ds, 10
    
    elif dataset_name == "mnist":
        transform = torchvision.transforms.Compose(
            [
                resize_trasnform,
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        train_ds = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform
        )

        valid_ds = torchvision.datasets.MNIST(
            root=data_dir, train=False, download=True, transform=transform
        )

        return train_ds, valid_ds, 10
    
    
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented.")


def _split_dataset_to_iid(dataset, clients):
    parts = [len(dataset) // clients for _ in range(clients)]
    parts[0] += len(dataset) % clients
    print(f"Spliting Datasets {len(dataset)} into parts:{parts}")
    subsets = torch.utils.data.random_split(dataset, parts)
    return [SubsetToDataset(subset) for subset in subsets]


def _split_dataset_to_niid(dataset, clients, beta=3):
    assert beta > 0 and beta < 4, "beta must be in (0,4)"
    min_data = (len(dataset) // clients) // beta
    max_data = len(dataset) // clients
    parts = [random.randint(min_data, max_data) for _ in range(clients)]
    remaining_data = len(dataset) - sum(parts)
    i = 0
    while remaining_data > 0:
        parts[i] += 1
        remaining_data -= 1
        i += 1
        if i == len(parts):
            i = 0

    assert sum(parts) == len(
        dataset
    ), f"Sum of parts is not equal to dataset size: {sum(parts)} != {len(dataset)}"

    print(f"Spliting Datasets {len(dataset)} into parts:{parts}")
    subsets = torch.utils.data.random_split(dataset, parts)
    clients_datasets = [SubsetToDataset(subset) for subset in subsets]

    assert sum(parts) == sum(
        (len(d) for d in clients_datasets)
    ), f"Sum of parts is not equal to dataset size: \
        /{sum(parts)} != {sum([len(dataset) for dataset in clients_datasets])}"

    return clients_datasets


class SubsetToDataset(torch.utils.data.Dataset):
    """Convert a Subset to a Dataset."""

    def __init__(self, subset, greyscale=False):
        self.subset = subset
        self.greyscale = greyscale

    def __getitem__(self, index):
        """Return the item at the given index."""
        x, y = self.subset[index]
        return x, y

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.subset)


def prepare_iid_dataset(dname, dataset_dir, num_clients):
    """Prepare the IID dataset."""
    train, valid, num_classes = _get_train_val_datasets(dname, data_dir=dataset_dir)
    clients_datasets = _split_dataset_to_iid(train, clients=num_clients)
    client2dataset = {f"{cid}": d for cid, d in enumerate(clients_datasets)}
    return client2dataset, valid, num_classes


def prepare_niid_dataset(dname, dataset_dir, num_clients):
    """Prepare the non-IID dataset."""
    train, valid, num_classes = _get_train_val_datasets(dname, data_dir=dataset_dir)
    clients_datasets = _split_dataset_to_niid(train, clients=num_clients)
    client2dataset = {f"{cid}": d for cid, d in enumerate(clients_datasets)}
    return client2dataset, valid, num_classes
