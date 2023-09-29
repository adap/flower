"""Handle basic dataset creation.

In case of PyTorch it should return dataloaders for your dataset (for both the clients
and the server). If you are using a custom dataset class, this module is the place to
define it. If your dataset requires to be downloaded (and this is not done
automatically -- e.g. as it is the case for many dataset in TorchVision) and
partitioned, please include all those functions and logic in the
`dataset_preparation.py` module. You can use all those functions from functions/methods
defined here of course.
"""

from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig
from typing import Optional, Tuple
from dataset_preparation import _partition_data, split_train_validation_test_clients
import numpy as np
import torchvision.transforms as transforms
from utils import word_to_indices, letter_to_vec
import torch


class ShakespeareDataset(Dataset):
    def __init__(self, data):
        sentence, label = data['x'], data['y']
        sentences_to_indices = [word_to_indices(word) for word in sentence]
        sentences_to_indices = np.array(sentences_to_indices)
        self.sentences_to_indices = np.array(sentences_to_indices, dtype=np.int64)
        self.labels = np.array([letter_to_vec(letter) for letter in label], dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data, target = self.sentences_to_indices[index], self.labels[index]
        return torch.tensor(data), torch.tensor(target)


class FemnistDataset(Dataset):
    def __init__(self, dataset, transform):
        self.x = dataset['x']
        self.y = dataset['y']
        self.transform = transform

    def __getitem__(self, index):
        input_data = np.array(self.x[index]).reshape(28, 28, 1)
        if self.transform:
            input_data = self.transform(input_data)
        target_data = self.y[index]
        return input_data.to(torch.float32), target_data

    def __len__(self):
        return len(self.y)


def load_datasets(  # pylint: disable=too-many-arguments
    config: DictConfig,
    path: str,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    dataset = _partition_data(
        data_type=config.data,
        dir_path=path,
        support_ratio=config.support_ratio
    )

    clients_list = split_train_validation_test_clients(
        dataset[0]['users']
    )

    trainloaders = {'sup': [], 'qry': []}
    valloaders = {'sup': [], 'qry': []}
    testloaders = {'sup': [], 'qry': []}

    data_type = config.data
    if data_type == 'femnist':
        transform = transforms.Compose([transforms.ToTensor()])
        for user in clients_list[0]:
            trainloaders['sup'].append(
                DataLoader(FemnistDataset(dataset[0]['user_data'][user], transform), batch_size=config.batch_size, shuffle=True))
            trainloaders['qry'].append(
                DataLoader(FemnistDataset(dataset[1]['user_data'][user], transform), batch_size=config.batch_size))
        for user in clients_list[1]:
            valloaders['sup'].append(
                DataLoader(FemnistDataset(dataset[0]['user_data'][user], transform), batch_size=config.batch_size))
            valloaders['qry'].append(
                DataLoader(FemnistDataset(dataset[1]['user_data'][user], transform), batch_size=config.batch_size))
        for user in clients_list[2]:
            testloaders['sup'].append(
                DataLoader(FemnistDataset(dataset[0]['user_data'][user], transform), batch_size=config.batch_size))
            testloaders['qry'].append(
                DataLoader(FemnistDataset(dataset[1]['user_data'][user], transform), batch_size=config.batch_size))

    elif data_type == 'shakespeare':
        for user in clients_list[0]:
            trainloaders['sup'].append(
                DataLoader(ShakespeareDataset(dataset[0]['user_data'][user]), batch_size=config.batch_size, shuffle=True))
            trainloaders['qry'].append(
                DataLoader(ShakespeareDataset(dataset[1]['user_data'][user]), batch_size=config.batch_size))
        for user in clients_list[1]:
            valloaders['sup'].append(
                DataLoader(ShakespeareDataset(dataset[0]['user_data'][user]), batch_size=config.batch_size, shuffle=True))
            valloaders['qry'].append(
                DataLoader(ShakespeareDataset(dataset[1]['user_data'][user]), batch_size=config.batch_size))
        for user in clients_list[2]:
            testloaders['sup'].append(
                DataLoader(ShakespeareDataset(dataset[0]['user_data'][user]), batch_size=config.batch_size, shuffle=True))
            testloaders['qry'].append(
                DataLoader(ShakespeareDataset(dataset[1]['user_data'][user]), batch_size=config.batch_size))

    return trainloaders, valloaders, testloaders



