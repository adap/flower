import os
import tarfile
from urllib import request

import numpy as np
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    ToTensor,
)


def _partition(files_list, labels_list, num_shards, index):
    total_size = len(files_list)
    assert total_size == len(
        labels_list
    ), f"List of datapoints and labels must be of the same length"
    shard_size = total_size // num_shards

    # Calculate start and end indices for the shard
    start_idx = index * shard_size
    if index == num_shards - 1:
        # Last shard takes the remainder
        end_idx = total_size
    else:
        end_idx = start_idx + shard_size

    # Create a subset for the shard
    files = files_list[start_idx:end_idx]
    labels = labels_list[start_idx:end_idx]
    return files, labels


def load_data(num_shards, index):
    image_file_list, image_label_list, _, num_class = _download_data()

    # Get partition given index
    files_list, labels_list = _partition(
        image_file_list, image_label_list, num_shards, index
    )

    trainX, trainY, valX, valY, testX, testY = _split_data(
        files_list, labels_list, len(files_list)
    )
    train_transforms, val_transforms = _get_transforms()

    train_ds = MedNISTDataset(trainX, trainY, train_transforms)
    train_loader = DataLoader(train_ds, batch_size=300, shuffle=True)

    val_ds = MedNISTDataset(valX, valY, val_transforms)
    val_loader = DataLoader(val_ds, batch_size=300)

    test_ds = MedNISTDataset(testX, testY, val_transforms)
    test_loader = DataLoader(test_ds, batch_size=300)

    return train_loader, val_loader, test_loader, num_class


class MedNISTDataset(Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]


def _download_data():
    data_dir = "./MedNIST/"
    _download_and_extract(
        "https://dl.dropboxusercontent.com/s/5wwskxctvcxiuea/MedNIST.tar.gz",
        os.path.join(data_dir),
    )

    class_names = sorted(
        [x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x))]
    )
    num_class = len(class_names)
    image_files = [
        [
            os.path.join(data_dir, class_name, x)
            for x in os.listdir(os.path.join(data_dir, class_name))
        ]
        for class_name in class_names
    ]
    image_file_list = []
    image_label_list = []
    for i, class_name in enumerate(class_names):
        image_file_list.extend(image_files[i])
        image_label_list.extend([i] * len(image_files[i]))
    num_total = len(image_label_list)
    return image_file_list, image_label_list, num_total, num_class


def _split_data(image_file_list, image_label_list, num_total):
    valid_frac, test_frac = 0.1, 0.1
    trainX, trainY = [], []
    valX, valY = [], []
    testX, testY = [], []

    for i in range(num_total):
        rann = np.random.random()
        if rann < valid_frac:
            valX.append(image_file_list[i])
            valY.append(image_label_list[i])
        elif rann < test_frac + valid_frac:
            testX.append(image_file_list[i])
            testY.append(image_label_list[i])
        else:
            trainX.append(image_file_list[i])
            trainY.append(image_label_list[i])

    return trainX, trainY, valX, valY, testX, testY


def _get_transforms():
    train_transforms = Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            ScaleIntensity(),
            RandRotate(range_x=15, prob=0.5, keep_size=True),
            RandFlip(spatial_axis=0, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
            ToTensor(),
        ]
    )

    val_transforms = Compose(
        [LoadImage(image_only=True), EnsureChannelFirst(), ScaleIntensity(), ToTensor()]
    )

    return train_transforms, val_transforms


def _download_and_extract(url, dest_folder):
    if not os.path.isdir(dest_folder):
        # Download the tar.gz file
        tar_gz_filename = url.split("/")[-1]
        if not os.path.isfile(tar_gz_filename):
            with request.urlopen(url) as response, open(
                tar_gz_filename, "wb"
            ) as out_file:
                out_file.write(response.read())

        # Extract the tar.gz file
        with tarfile.open(tar_gz_filename, "r:gz") as tar_ref:
            tar_ref.extractall()
