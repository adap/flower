"""CIFAR10 dataset class, adopted from authors implementation."""

import os
import pickle

import anytree
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from heterofl.datasets.utils import (
    download_url,
    extract_file,
    make_classes_counts,
    make_flat_index,
    make_tree,
)
from heterofl.utils import check_exists, load, makedir_exist_ok, save


# pylint: disable=too-many-instance-attributes
class CIFAR10(Dataset):
    """CIFAR10 dataset."""

    data_name = "CIFAR10"
    file = [
        (
            "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
            "c58f30108f718f92721af3b95e74349a",
        )
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
        img, target = Image.fromarray(self.img[index]), torch.tensor(self.target[index])
        inp = {"img": img, self.subset: target}
        if self.transform is not None:
            inp = self.transform(inp)
        return inp["img"], inp["label"]

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

    def download(self):
        """Download dataset from the url."""
        makedir_exist_ok(self.raw_folder)
        for url, md5 in self.file:
            filename = os.path.basename(url)
            download_url(url, self.raw_folder, filename, md5)
            extract_file(os.path.join(self.raw_folder, filename))

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
        train_filenames = [
            "data_batch_1",
            "data_batch_2",
            "data_batch_3",
            "data_batch_4",
            "data_batch_5",
        ]
        test_filenames = ["test_batch"]
        train_img, train_label = _read_pickle_file(
            os.path.join(self.raw_folder, "cifar-10-batches-py"), train_filenames
        )
        test_img, test_label = _read_pickle_file(
            os.path.join(self.raw_folder, "cifar-10-batches-py"), test_filenames
        )
        train_target, test_target = {"label": train_label}, {"label": test_label}
        with open(
            os.path.join(self.raw_folder, "cifar-10-batches-py", "batches.meta"), "rb"
        ) as fle:
            data = pickle.load(fle, encoding="latin1")
            classes = data["label_names"]
        classes_to_labels = {"label": anytree.Node("U", index=[])}
        for cls in classes:
            make_tree(classes_to_labels["label"], [cls])
        classes_size = {"label": make_flat_index(classes_to_labels["label"])}
        return (
            (train_img, train_target),
            (test_img, test_target),
            (classes_to_labels, classes_size),
        )


def _read_pickle_file(path, filenames):
    img, label = [], []
    for filename in filenames:
        file_path = os.path.join(path, filename)
        with open(file_path, "rb") as file:
            entry = pickle.load(file, encoding="latin1")
            img.append(entry["data"])
            if "labels" in entry:
                label.extend(entry["labels"])
            else:
                label.extend(entry["fine_labels"])
            # label.extend(entry["labels"]) if "labels" in entry else label.extend(
            #     entry["fine_labels"]
            # )
    img = np.vstack(img).reshape(-1, 3, 32, 32)
    img = img.transpose((0, 2, 3, 1))
    return img, label
