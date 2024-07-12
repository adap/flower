"""Dataset preparation."""

import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms


class BaseDataset(Dataset):
    """Base class for all datasets."""

    def __init__(
        self,
        root: Path = Path("datasets/cifar10"),
        train_data_transform: transforms.transforms.Compose = None,
        test_data_transform: transforms.transforms.Compose = None,
    ) -> None:
        """Initialize the dataset."""
        self.root = root
        self.classes = None
        self.data: torch.tensor = None
        self.targets: torch.tensor = None
        self.train_data_transform = train_data_transform
        self.test_data_transform = test_data_transform
        self.training = True

    def train(self):
        """Activate data transformation for training."""
        self.training = True

    def eval(self):
        """Activate data transformation for testing."""
        self.training = False

    def __getitem__(self, index):
        """Get the item at the given index."""
        data, targets = self.data[index], self.targets[index]
        data_transform = (
            self.train_data_transform if self.training else self.test_data_transform
        )
        if data_transform is not None:
            data = data_transform(data)
        return data, targets

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.targets)


class CIFAR10(BaseDataset):
    """CIFAR10 dataset."""

    def __init__(
        self,
        root: Path = Path("datasets/cifar10"),
        train_data_transform: transforms.transforms.Compose = None,
        test_data_transform: transforms.transforms.Compose = None,
    ):
        super().__init__()

        train_part = torchvision.datasets.CIFAR10(root, True, download=True)
        test_part = torchvision.datasets.CIFAR10(root, False, download=True)
        train_data = torch.tensor(train_part.data).permute([0, -1, 1, 2]).float()
        test_data = torch.tensor(test_part.data).permute([0, -1, 1, 2]).float()
        train_targets = torch.tensor(train_part.targets).long().squeeze()
        test_targets = torch.tensor(test_part.targets).long().squeeze()
        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = train_part.classes
        self.train_data_transform = train_data_transform
        self.test_data_transform = test_data_transform


class CIFAR100(BaseDataset):
    """CIFAR100 dataset."""

    def __init__(
        self,
        root: Path = Path("datasets/cifar100"),
        train_data_transform: transforms.transforms.Compose = None,
        test_data_transform: transforms.transforms.Compose = None,
    ):
        super().__init__()
        train_part = torchvision.datasets.CIFAR100(root, True, download=True)
        test_part = torchvision.datasets.CIFAR100(root, False, download=True)
        train_data = torch.tensor(train_part.data).permute([0, -1, 1, 2]).float()
        test_data = torch.tensor(test_part.data).permute([0, -1, 1, 2]).float()
        train_targets = torch.tensor(train_part.targets).long().squeeze()
        test_targets = torch.tensor(test_part.targets).long().squeeze()
        self.data = torch.cat([train_data, test_data])
        self.targets = torch.cat([train_targets, test_targets])
        self.classes = train_part.classes
        self.train_data_transform = train_data_transform
        self.test_data_transform = test_data_transform


def call_dataset(dataset_name, root, **kwargs):
    """Call the dataset."""
    if dataset_name == "cifar10":
        return CIFAR10(root, **kwargs)
    if dataset_name == "cifar100":
        return CIFAR100(root, **kwargs)
    raise ValueError(f"Dataset {dataset_name} not supported.")


def randomly_assign_classes(
    dataset: Dataset, client_num: int, class_num: int
) -> Dict[str, Union[Dict[Any, Any], List[Any]]]:
    # ) -> Dict[str, Any]:
    """Randomly assign number classes to clients."""
    partition: Dict[str, Union[Dict, List]] = {"separation": {}, "data_indices": []}
    data_indices: List[List[int]] = [[] for _ in range(client_num)]
    targets_numpy = np.array(dataset.targets, dtype=np.int32)
    label_list = list(range(len(dataset.classes)))

    data_idx_for_each_label = [
        np.where(targets_numpy == i)[0].tolist() for i in label_list
    ]

    assigned_labels = []
    selected_times = [0 for _ in label_list]
    for _ in range(client_num):
        sampled_labels = random.sample(label_list, class_num)
        assigned_labels.append(sampled_labels)
        for j in sampled_labels:
            selected_times[j] += 1

    batch_sizes = _get_batch_sizes(
        targets_numpy=targets_numpy,
        label_list=label_list,
        selected_times=selected_times,
    )

    data_indices = _get_data_indices(
        batch_sizes=batch_sizes,
        data_indices=data_indices,
        data_idx_for_each_label=data_idx_for_each_label,
        assigned_labels=assigned_labels,
        client_num=client_num,
    )

    partition["data_indices"] = data_indices

    return partition  # , stats


def _get_batch_sizes(
    targets_numpy: np.ndarray, label_list: List[int], selected_times: List[int]
) -> np.ndarray:
    """Get batch sizes for each label."""
    labels_count = Counter(targets_numpy)
    batch_sizes = np.zeros_like(label_list)
    for i in label_list:
        print(f"label: {i}, count: {labels_count[i]}")
        print(f"selected times: {selected_times[i]}")
        batch_sizes[i] = int(labels_count[i] / selected_times[i])

    return batch_sizes


def _get_data_indices(
    batch_sizes: np.ndarray,
    data_indices: List[List[int]],
    data_idx_for_each_label: List[List[int]],
    assigned_labels: List[List[int]],
    client_num: int,
) -> List[List[int]]:
    for i in range(client_num):
        for cls in assigned_labels[i]:
            if len(data_idx_for_each_label[cls]) < 2 * batch_sizes[cls]:
                batch_size = len(data_idx_for_each_label[cls])
            else:
                batch_size = batch_sizes[cls]
            selected_idx = random.sample(data_idx_for_each_label[cls], batch_size)
            data_indices_use: np.ndarray = np.concatenate(
                [data_indices[i], selected_idx], axis=0
            ).astype(np.int64)
            data_indices[i] = data_indices_use.tolist()
            # data_indices[i]: np.ndarray = np.concatenate(
            #    [data_indices[i], selected_idx], axis=0
            # ).astype(np.int64)
            data_idx_for_each_label[cls] = list(
                set(data_idx_for_each_label[cls]) - set(selected_idx)
            )

        data_indices[i] = data_indices[i]

    return data_indices
