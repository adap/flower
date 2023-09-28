from typing import Optional, Tuple, List
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader, random_split
from niid_bench.dataset_preparation import partition_data, partition_data_dirichlet, partition_data_label_quantity

def load_datasets(
    config: DictConfig,
    num_clients: int,
    val_ratio: float = 0.1,
    seed: Optional[int] = 42,
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    """Creates the dataloaders to be fed into the model.

    Parameters
    ----------
    config: DictConfig
        Parameterises the dataset partitioning process
    num_clients : int
        The number of clients that hold a part of the data
    val_ratio : float, optional
        The ratio of training data that will be used for validation (between 0 and 1),
        by default 0.1
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        The DataLoader for training, the DataLoader for validation, the DataLoader for testing.
    """
    print(f"Dataset partitioning config: {config}")
    partitioning = ""
    if "_partitioning" in config:
        partitioning = config._partitioning
    # partition the data
    if partitioning == "dirichlet":
        alpha = 0.5
        if "alpha" in config:
            alpha = config.alpha
        datasets, testset = partition_data_dirichlet(
            num_clients,
            alpha=alpha,
            seed=seed,
            dataset_name=config._name,
        )
    elif partitioning == "label_quantity":
        labels_per_client = 2
        if "labels_per_client" in config:
            labels_per_client = config.labels_per_client
        datasets, testset = partition_data_label_quantity(
            num_clients,
            labels_per_client=labels_per_client,
            seed=seed,
            dataset_name=config._name,
        )
    elif partitioning == "iid":
        datasets, testset = partition_data(
            num_clients,
            similarity=1.0,
            seed=seed,
            dataset_name=config._name,
        )
    elif partitioning == "iid_noniid":
        similarity = 0.5
        if "similarity" in config:
            similarity = config.similarity
        datasets, testset = partition_data(
            num_clients,
            similarity=similarity,
            seed=seed,
            dataset_name=config._name,
        )

    batch_size = -1
    if "batch_size" in config:
        batch_size = config.batch_size
    elif "batch_size_ratio" in config:
        batch_size_ratio = config.batch_size_ratio
    else:
        raise ValueError

    # split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for dataset in datasets:
        len_val = int(len(dataset) / (1 / val_ratio)) if val_ratio > 0 else 0
        lengths = [len(dataset) - len_val, len_val]
        ds_train, ds_val = random_split(
            dataset, lengths, torch.Generator().manual_seed(seed)
        )
        if batch_size == -1:
            batch_size = int(len(ds_train) * batch_size_ratio)
        trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=batch_size))
    return trainloaders, valloaders, DataLoader(testset, batch_size=len(testset))