"""MNIST dataset class."""
import codecs
import os

import anytree
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import check_exists, load, makedir_exist_ok, save

from .utils import (
    download_url,
    extract_file,
    make_classes_counts,
    make_flat_index,
    make_tree,
)


class MNIST(Dataset):
    """MNIST dataset."""

    data_name = "MNIST"
    file = [
        (
            "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
            "f68b3c2dcbeaaa9fbdd348bbdeb94873",
        ),
        (
            "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
            "9fb629c4189551a2d022fa330f9573f3",
        ),
        (
            "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
            "d53e105ee54ea40749a09fcbcd1e9432",
        ),
        (
            "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
            "ec29112dd5afa0611ce80d1b7f02629c",
        ),
    ]

    def __init__(self, root, split, subset, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.subset = subset
        self.transform = transform
        if not check_exists(self.processed_folder):
            self.process()
        self.img, self.target = load(
            os.path.join(self.processed_folder, "{}.pt".format(self.split))
        )
        self.target = self.target[self.subset]
        self.classes_counts = make_classes_counts(self.target)
        self.classes_to_labels, self.classes_size = load(
            os.path.join(self.processed_folder, "meta.pt")
        )
        self.classes_to_labels, self.classes_size = (
            self.classes_to_labels[self.subset],
            self.classes_size[self.subset],
        )

    def __getitem__(self, index):
        """Get the item with index."""
        img, target = Image.fromarray(self.img[index], mode="L"), torch.tensor(
            self.target[index]
        )
        input = {"img": img, self.subset: target}
        if self.transform is not None:
            input = self.transform(input)
        return input

    def __len__(self):
        """Length of the dataset."""
        return len(self.img)

    @property
    def processed_folder(self):
        """Return path of processed folder."""
        return os.path.join(self.root, "processed")

    @property
    def raw_folder(self):
        """Return path of raw folder."""
        return os.path.join(self.root, "raw")

    def process(self):
        """Save the dataset accordingly."""
        if not check_exists(self.raw_folder):
            self.download()
        train_set, test_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, "train.pt"))
        save(test_set, os.path.join(self.processed_folder, "test.pt"))
        save(meta, os.path.join(self.processed_folder, "meta.pt"))
        return

    def download(self):
        """Download and save the dataset accordingly."""
        makedir_exist_ok(self.raw_folder)
        for url, md5 in self.file:
            filename = os.path.basename(url)
            download_url(url, self.raw_folder, filename, md5)
            extract_file(os.path.join(self.raw_folder, filename))
        return

    def __repr__(self):
        """Represent CIFAR10 as string."""
        fmt_str = (
            f"Dataset {self.__class__.__name__}\nSize: {self.__len__()}\n"
            f"Root: {self.root}\nSplit: {self.split}\nSubset: {self.subset}\n"
            f"Transforms: {self.transform.__repr__()}"
        )
        return fmt_str

    def make_data(self):
        """Make data."""
        train_img = _read_image_file(
            os.path.join(self.raw_folder, "train-images-idx3-ubyte")
        )
        test_img = _read_image_file(
            os.path.join(self.raw_folder, "t10k-images-idx3-ubyte")
        )
        train_label = _read_label_file(
            os.path.join(self.raw_folder, "train-labels-idx1-ubyte")
        )
        test_label = _read_label_file(
            os.path.join(self.raw_folder, "t10k-labels-idx1-ubyte")
        )
        train_target, test_target = {"label": train_label}, {"label": test_label}
        classes_to_labels = {"label": anytree.Node("U", index=[])}
        classes = list(map(str, list(range(10))))
        for c in classes:
            make_tree(classes_to_labels["label"], [c])
        classes_size = {"label": make_flat_index(classes_to_labels["label"])}
        return (
            (train_img, train_target),
            (test_img, test_target),
            (classes_to_labels, classes_size),
        )


def _get_int(b):
    return int(codecs.encode(b, "hex"), 16)


def _read_image_file(path):
    with open(path, "rb") as f:
        data = f.read()
        assert _get_int(data[:4]) == 2051
        length = _get_int(data[4:8])
        num_rows = _get_int(data[8:12])
        num_cols = _get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16).reshape(
            (length, num_rows, num_cols)
        )
        return parsed


def _read_label_file(path):
    with open(path, "rb") as f:
        data = f.read()
        assert _get_int(data[:4]) == 2049
        length = _get_int(data[4:8])
        parsed = (
            np.frombuffer(data, dtype=np.uint8, offset=8)
            .reshape(length)
            .astype(np.int64)
        )
        return parsed
