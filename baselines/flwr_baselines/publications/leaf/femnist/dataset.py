from typing import List

import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import numpy as np
import torch
from sklearn import preprocessing


class NISTLikeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=transforms.ToTensor(), target_transform=transforms.ToTensor()):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(
            image_path)  # todo: verify if .convert("L") is not needed since images are already in grayscale
        if self.transform:
            image = self.transform(image)
            label = torch.tensor(label)
        return image, label


def create_dataset(df_info: pd.DataFrame, labels):
    nist_like_dataset = NISTLikeDataset(df_info["path"].values, labels)
    return nist_like_dataset


def create_division_list(df_info: pd.DataFrame):
    writers_ids = df_info["writer_id"].values
    unique_writers = np.unique(writers_ids)
    indices = {writer_id: np.where(writers_ids == writer_id)[0].tolist() for writer_id in unique_writers}
    return list(indices.values())


def partition_dataset(dataset: Dataset, division_list) -> List[Dataset]:
    subsets = []
    for sequence in division_list:
        subsets.append(Subset(dataset, sequence))
    return subsets


def get_partitioned_train_test_dataset(partitioned_dataset, train_split: float = 0.9, test_split: float = 0.1):
    # todo: remove redundant test_split_param (or add also validation argument)
    train_subsets = []
    test_subsets = []

    for subset in partitioned_dataset:
        subset_len = len(subset)
        train_len = int(train_split * subset_len)
        test_len = subset_len - train_len
        train_dataset, test_dataset = random_split(
            subset,
            lengths=[train_len, test_len],
            generator=torch.Generator().manual_seed(42))
        train_subsets.append(train_dataset)
        test_subsets.append(test_dataset)

    return train_subsets, test_subsets


def transform_datasets_into_dataloaders(datasets, **kwargs):
    """kwargs covers mostly batch_size, and shuffle, think of all the arguments that the DataLoader takes"""
    dataloaders = []
    for dataset in datasets:
        dataloaders.append(DataLoader(dataset, **kwargs))
    return dataloaders


if __name__ == "__main__":
    sampled_data_info = pd.read_csv("data/processed/niid_sampled_images_to_labels.csv")
    label_encoder = preprocessing.LabelEncoder()
    labels = label_encoder.fit_transform(sampled_data_info["character"])
    # Create a list of DataLoaders
    full_dataset = create_dataset(sampled_data_info, labels)
    division_list = create_division_list(sampled_data_info)
    partitioned_dataset = partition_dataset(full_dataset, division_list)
    partitioned_train, partitioned_test = get_partitioned_train_test_dataset(partitioned_dataset)
    trainloaders = transform_datasets_into_dataloaders(partitioned_train)
    testloaders = transform_datasets_into_dataloaders(partitioned_test)
    input()
