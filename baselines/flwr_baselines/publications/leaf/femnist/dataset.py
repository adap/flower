from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset, Subset, random_split


class NISTLikeDataset(Dataset):
    """Dataset representing NIST or preprocessed variant of it."""

    def __init__(self, image_paths, labels, transform=transforms.ToTensor()):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
            label = torch.tensor(label)
        return image, label


def create_dataset(df_info: pd.DataFrame, labels: np.ndarray):
    """Instantiate NISTLikeDataset.

    Parameters
    ----------
    df_info: pd.DataFrame
        contains paths to images
    labels: np.ndarray
        0 till N-1 classes labels in the same order as in df_info

    Returns
    -------
    nist_like_dataset: NISTLikeDataset
        created dataset
    """
    nist_like_dataset = NISTLikeDataset(df_info["path"].values, labels)
    return nist_like_dataset


def create_division_list(df_info: pd.DataFrame) -> List[List[int]]:
    """
    Create list of list with int masks identifying writers.
    Parameters
    ----------
    df_info: pd.DataFrame
        contains writer_id information

    Returns
    -------
    division_list: List[List[int]]
        List of lists of indices to identify unique writers
    """
    writers_ids = df_info["writer_id"].values
    unique_writers = np.unique(writers_ids)
    indices = {
        writer_id: np.where(writers_ids == writer_id)[0].tolist()
        for writer_id in unique_writers
    }
    return list(indices.values())


def partition_dataset(
    dataset: Dataset, division_list: List[List[int]]
) -> List[Dataset]:
    """
    Partition dataset for niid settings - by writer id (each partition has only single writer data).
    Parameters
    ----------
    dataset: Dataset
        dataset of all images
    division_list: List[List[int]]
        list of lists of indices to identify unique writers

    Returns
    -------
    subsets: List[Dataset]
        subsets of datasets divided by writer id
    """
    subsets = []
    for sequence in division_list:
        subsets.append(Subset(dataset, sequence))
    return subsets


def partition_datasets(
    partitioned_dataset: List[Dataset],
    train_split: float = 0.8,
    validation_split: float = 0.1,
) -> Tuple[List[Dataset], List[Dataset], List[Dataset]]:
    """Partition list of datasets to train, validation and test splits (each
    dataset from the list individually).

    Parameters
    ----------
    partitioned_dataset: List[Dataset]
        partitioned datasets
    train_split: float
        part of the data used for training
    validation_split: float
        part of the data used for validation

    Returns
    -------
        (train, validation, test): Tuple[List[Dataset], List[Dataset], List[Dataset]]
        split datasets
    """
    train_subsets = []
    validation_subsets = []
    test_subsets = []

    for subset in partitioned_dataset:
        subset_len = len(subset)
        train_len = int(train_split * subset_len)
        val_len = int(validation_split * subset_len)
        test_len = subset_len - train_len - val_len
        train_dataset, validation_dataset, test_dataset = random_split(
            subset,
            lengths=[train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(42),
        )
        train_subsets.append(train_dataset)
        validation_subsets.append(validation_dataset)
        test_subsets.append(test_dataset)

    return train_subsets, validation_subsets, test_subsets


def transform_datasets_into_dataloaders(
    datasets: List[Dataset], **dataloader_kwargs
) -> List[DataLoader]:
    """
    Transform datasets into dataloaders.
    Parameters
    ----------
    datasets: List[Dataset]
        list of datasets
    dataloader_kwargs
        arguments to DataLoader

    Returns
    -------
    dataloader: List[DataLoader]
        list of dataloaders
    """
    dataloaders = []
    for dataset in datasets:
        dataloaders.append(DataLoader(dataset, **dataloader_kwargs))
    return dataloaders


if __name__ == "__main__":
    sampled_data_info = pd.read_csv("data/processed/niid_sampled_images_to_labels.csv")
    label_encoder = preprocessing.LabelEncoder()
    labels = label_encoder.fit_transform(sampled_data_info["character"])
    # Create a list of DataLoaders
    full_dataset = create_dataset(sampled_data_info, labels)
    division_list = create_division_list(sampled_data_info)
    # Partitioned by writer (therefore by client in the FL settings)
    partitioned_dataset = partition_dataset(full_dataset, division_list)
    partitioned_train, partitioned_validation, partitioned_test = partition_datasets(
        partitioned_dataset
    )
    trainloaders = transform_datasets_into_dataloaders(partitioned_train)
    testloaders = transform_datasets_into_dataloaders(partitioned_test)
