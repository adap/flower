"""Handle basic dataset creation.

In case of PyTorch it should return dataloaders for your dataset (for both the clients
and the server). If you are using a custom dataset class, this module is the place to
define it. If your dataset requires to be downloaded (and this is not done
automatically -- e.g. as it is the case for many dataset in TorchVision) and
partitioned, please include all those functions and logic in the
`dataset_preparation.py` module. You can use all those functions from functions/methods
defined here of course.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from fedmeta.dataset_preparation import (
    _partition_data,
    split_train_validation_test_clients,
)
from fedmeta.utils import letter_to_vec, word_to_indices


class ShakespeareDataset(Dataset):
    """
    [LEAF: A Benchmark for Federated Settings](https://github.com/TalwalkarLab/leaf).

    We imported the preprocessing method for the Shakespeare dataset from GitHub.

    word_to_indices : returns a list of character indices
    sentences_to_indices: converts an index to a one-hot vector of a given size.
    letter_to_vec : returns one-hot representation of given letter

    """

    def __init__(self, data):
        sentence, label = data["x"], data["y"]
        sentences_to_indices = [word_to_indices(word) for word in sentence]
        sentences_to_indices = np.array(sentences_to_indices)
        self.sentences_to_indices = np.array(sentences_to_indices, dtype=np.int64)
        self.labels = np.array(
            [letter_to_vec(letter) for letter in label], dtype=np.int64
        )

    def __len__(self):
        """Return the number of labels present in the dataset.

        Returns
        -------
            int: The total number of labels.
        """
        return len(self.labels)

    def __getitem__(self, index):
        """Retrieve the data and its corresponding label at a given index.

        Args:
            index (int): The index of the data item to fetch.

        Returns
        -------
            tuple: (data tensor, label tensor)
        """
        data, target = self.sentences_to_indices[index], self.labels[index]
        return torch.tensor(data), torch.tensor(target)


class FemnistDataset(Dataset):
    """
    [LEAF: A Benchmark for Federated Settings](https://github.com/TalwalkarLab/leaf).

    We imported the preprocessing method for the Femnist dataset from GitHub.
    """

    def __init__(self, dataset, transform):
        self.x = dataset["x"]
        self.y = dataset["y"]
        self.transform = transform

    def __getitem__(self, index):
        """Retrieve the input data and its corresponding label at a given index.

        Args:
            index (int): The index of the data item to fetch.

        Returns
        -------
            tuple:
                - input_data (torch.Tensor): Reshaped and optionally transformed data.
                - target_data (int or torch.Tensor): Label for the input data.
        """
        input_data = np.array(self.x[index]).reshape(28, 28)
        if self.transform:
            input_data = self.transform(input_data)
        target_data = self.y[index]
        return input_data.to(torch.float32), target_data

    def __len__(self):
        """Return the number of labels present in the dataset.

        Returns
        -------
            int: The total number of labels.
        """
        return len(self.y)


def load_datasets(
    config: DictConfig,
    path: str,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create the dataloaders to be fed into the model.

    Parameters
    ----------
    config: DictConfig
        data: float
            Used data type
        batch_size : int
            The size of the batches to be fed into the model,
            by default 10
        support_ratio : float
            The ratio of Support set for each client.(between 0 and 1)
            by default 0.2
    path : str
        The path where the leaf dataset was downloaded

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
    """
    dataset = _partition_data(
        data_type=config.data, dir_path=path, support_ratio=config.support_ratio
    )

    # Client list : 0.8, 0.1, 0.1
    clients_list = split_train_validation_test_clients(dataset[0]["users"])

    trainloaders: Dict[str, List[DataLoader]] = {"sup": [], "qry": []}
    valloaders: Dict[str, List[DataLoader]] = {"sup": [], "qry": []}
    testloaders: Dict[str, List[DataLoader]] = {"sup": [], "qry": []}

    data_type = config.data
    if data_type == "femnist":
        transform = transforms.Compose([transforms.ToTensor()])
        for user in clients_list[0]:
            trainloaders["sup"].append(
                DataLoader(
                    FemnistDataset(dataset[0]["user_data"][user], transform),
                    batch_size=config.batch_size,
                    shuffle=True,
                )
            )
            trainloaders["qry"].append(
                DataLoader(
                    FemnistDataset(dataset[1]["user_data"][user], transform),
                    batch_size=config.batch_size,
                )
            )
        for user in clients_list[1]:
            valloaders["sup"].append(
                DataLoader(
                    FemnistDataset(dataset[0]["user_data"][user], transform),
                    batch_size=config.batch_size,
                )
            )
            valloaders["qry"].append(
                DataLoader(
                    FemnistDataset(dataset[1]["user_data"][user], transform),
                    batch_size=config.batch_size,
                )
            )
        for user in clients_list[2]:
            testloaders["sup"].append(
                DataLoader(
                    FemnistDataset(dataset[0]["user_data"][user], transform),
                    batch_size=config.batch_size,
                )
            )
            testloaders["qry"].append(
                DataLoader(
                    FemnistDataset(dataset[1]["user_data"][user], transform),
                    batch_size=config.batch_size,
                )
            )

    elif data_type == "shakespeare":
        for user in clients_list[0]:
            trainloaders["sup"].append(
                DataLoader(
                    ShakespeareDataset(dataset[0]["user_data"][user]),
                    batch_size=config.batch_size,
                    shuffle=True,
                )
            )
            trainloaders["qry"].append(
                DataLoader(
                    ShakespeareDataset(dataset[1]["user_data"][user]),
                    batch_size=config.batch_size,
                )
            )
        for user in clients_list[1]:
            valloaders["sup"].append(
                DataLoader(
                    ShakespeareDataset(dataset[0]["user_data"][user]),
                    batch_size=config.batch_size,
                    shuffle=True,
                )
            )
            valloaders["qry"].append(
                DataLoader(
                    ShakespeareDataset(dataset[1]["user_data"][user]),
                    batch_size=config.batch_size,
                )
            )
        for user in clients_list[2]:
            testloaders["sup"].append(
                DataLoader(
                    ShakespeareDataset(dataset[0]["user_data"][user]),
                    batch_size=config.batch_size,
                    shuffle=True,
                )
            )
            testloaders["qry"].append(
                DataLoader(
                    ShakespeareDataset(dataset[1]["user_data"][user]),
                    batch_size=config.batch_size,
                )
            )

    return trainloaders, valloaders, testloaders
