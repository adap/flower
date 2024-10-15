
import random
from functools import partial
from typing import Any, Callable, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from flwr.common.logger import log
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, PathologicalPartitioner
from logging import INFO, WARNING


class ClientsAndServerDatasets:
    """Prepare the clients and server datasets for federated learning."""

    def __init__(self, cfg: Any):
        """
        Initialize the dataset preparation.

        Args:
            cfg: Configuration object containing dataset and distribution parameters.
        """
        self.cfg = cfg
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.client_id_to_loader: Dict[str, DataLoader] = {}
        self.server_testloader: DataLoader = None

        self._set_distribution_partitioner()
        self._load_datasets()
        self._introduce_label_noise()

    def _set_distribution_partitioner(self):
        """Set the data distribution partitioner based on configuration."""
        dist_type = self.cfg.data_dist.dist_type
        if dist_type == 'non_iid_dirichlet':
            self.data_dist_partitioner_func = self._dirichlet_data_distribution
        elif dist_type.startswith('PathologicalPartitioner-'):
            try:
                num_classes = int(dist_type.split('-')[-1])
            except ValueError:
                raise ValueError(f"Invalid number of classes in distribution type: {dist_type}")
            self.data_dist_partitioner_func = partial(self._pathological_partitioner, num_classes)
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")

    def _dirichlet_data_distribution(self, target_label_col: str = "label") -> Dict[str, Any]:
        """Partition data using Dirichlet distribution."""
        partitioner = DirichletPartitioner(
            num_partitions=self.cfg.num_clients,
            partition_by=target_label_col,
            alpha=self.cfg.dirichlet_alpha,
            min_partition_size=0,
            self_balancing=True,
            shuffle=True,
        )
        return self._partition_helper(partitioner)

    def _pathological_partitioner(self, num_classes_per_partition: int, target_label_col: str = "label") -> Dict[str, Any]:
        """Partition data using Pathological partitioner."""
        partitioner = PathologicalPartitioner(
            num_partitions=self.cfg.num_clients,
            partition_by=target_label_col,
            num_classes_per_partition=num_classes_per_partition,
            shuffle=True,
            class_assignment_mode='deterministic'
        )
        return self._partition_helper(partitioner)

    def _partition_helper(self, partitioner: Any) -> Dict[str, Any]:
        """
        Helper function to partition the dataset using the specified partitioner.

        Args:
            partitioner: The partitioner instance to use.

        Returns:
            A dictionary with 'client2data' and 'server_data'.
        """
        fds = FederatedDataset(dataset=self.cfg.dname, partitioners={"train": partitioner})
        server_data = fds.load_split("test")
        return {'client2data': fds, 'server_data': server_data}

    def _load_datasets(self):
        """Load and partition the datasets based on the partitioner."""
        supported_datasets = {"cifar10", "mnist"}  # Extend as needed
        if self.cfg.dname.lower() not in supported_datasets:
            raise ValueError(f"Dataset '{self.cfg.dname}' not supported.")

        partitions = self.data_dist_partitioner_func(self.cfg)

        # Apply transformations using with_transform
        self.client_id_to_loader = {
            client_id: self._create_dataloader(client_data)
            for client_id, client_data in partitions['client2data'].items()
        }
        self.server_testloader = self._create_dataloader(partitions['server_data'], batch_size=256, shuffle=False)

    def _create_dataloader(self, dataset: Any, batch_size: int = 64, shuffle: bool = True) -> DataLoader:
        """
        Create a DataLoader with applied transformations.

        Args:
            dataset: The dataset to wrap in a DataLoader.
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle the data.

        Returns:
            A PyTorch DataLoader instance.
        """
        # Define the transformation function
        def apply_transforms(batch: Dict[str, Any]) -> Dict[str, Any]:
            batch["img"] = [self.transform(img) for img in batch["img"]]
            return batch

        # Apply transformations on-the-fly
        transformed_dataset = dataset.with_transform(apply_transforms)

        # Create DataLoader
        dataloader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader

    def _introduce_label_noise(self):
        """Introduce label noise to specified faulty clients."""
        faulty_client_ids = self.cfg.faulty_clients_ids
        noise_rate = self.cfg.noise_rate
        num_classes = self.cfg.dataset.num_classes

        if faulty_client_ids:
            for client_id in faulty_client_ids:
                if client_id in self.client_id_to_loader:
                    # Modify the underlying dataset to introduce noise
                    dataset = self.client_id_to_loader[client_id].dataset
                    noisy_dataset = self._add_noise_in_data(
                        dataset=dataset,
                        noise_rate=noise_rate,
                        num_classes=num_classes
                    )
                    # Recreate DataLoader with the noisy dataset
                    self.client_id_to_loader[client_id] = DataLoader(noisy_dataset.with_transform(self._create_transform()),
                                                                      batch_size=self.client_id_to_loader[client_id].batch_size,
                                                                      shuffle=True)
                    log(
                        WARNING,
                        f"************* Client {client_id} is made noisy. *************",
                    )
                else:
                    log(
                        WARNING,
                        f"Client ID {client_id} not found in client data. Skipping noise addition.",
                    )
            log(
                INFO,
                f"** All Malicious Clients are: {faulty_client_ids} **",
            )

    def _create_transform(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """
        Create a transformation function to be used with with_transform.

        Returns:
            A callable that applies the necessary transformations to a batch.
        """
        def transform(batch: Dict[str, Any]) -> Dict[str, Any]:
            batch["img"] = [self.transform(img) for img in batch["img"]]
            return batch
        return transform

    def _add_noise_in_data(self, dataset: Any, noise_rate: float, num_classes: int) -> Any:
        """
        Introduce label noise by flipping labels based on the noise rate.

        Args:
            dataset: The dataset to modify.
            noise_rate: Probability of flipping each label (0 <= noise_rate <= 1).
            num_classes: Total number of classes in the dataset.

        Returns:
            The modified dataset with noisy labels.
        """
        def flip_labels(batch: Dict[str, Any]) -> Dict[str, Any]:
            labels = np.array(batch['label'])
            num_samples = len(labels)

            # Determine which labels to flip based on noise_rate
            flip_mask = np.random.rand(num_samples) < noise_rate
            indices_to_flip = np.where(flip_mask)[0]

            if len(indices_to_flip) == 0:
                return batch  # No labels to flip

            # Generate new labels ensuring they differ from current labels
            new_labels = labels[indices_to_flip].copy()
            for idx in indices_to_flip:
                current_label = new_labels[idx]
                possible_labels = list(range(num_classes))
                possible_labels.remove(current_label)
                new_labels[idx] = random.choice(possible_labels)

            # Assign the new labels back to the batch
            labels[indices_to_flip] = new_labels
            batch['label'] = labels.tolist()
            return batch

        # Apply the label flipping function to the dataset in batches
        noisy_dataset = dataset.map(
            flip_labels,
            batched=True,
            batch_size=256,
            num_proc=8
        )
        return noisy_dataset

    def get_data(self) -> Dict[str, Any]:
        """
        Get the prepared client and server DataLoaders.

        Returns:
            A dictionary containing 'server_testloader' and 'client_id_to_loader'.
        """
        return {
            "server_testloader": self.server_testloader,
            "client_id_to_loader": self.client_id_to_loader,
        }


def get_clients_server_data(cfg: Any) -> Dict[str, Any]:
    """
    Prepare and retrieve the clients and server DataLoaders.

    Args:
        cfg: Configuration object containing dataset and distribution parameters.

    Returns:
        A dictionary containing 'server_testloader' and 'client_id_to_loader'.
    """
    ds_prep = ClientsAndServerDatasets(cfg)
    return ds_prep.get_data()
