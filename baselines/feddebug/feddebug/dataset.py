"""Handle basic dataset creation.

In case of PyTorch it should return dataloaders for your dataset (for both the clients
and the server). If you are using a custom dataset class, this module is the place to
define it. If your dataset requires to be downloaded (and this is not done
automatically -- e.g. as it is the case for many dataset in TorchVision) and
partitioned, please include all those functions and logic in the
`dataset_preparation.py` module. You can use all those functions from functions/methods
defined here of course.
"""

import random
import numpy as np
from torch.utils.data import DataLoader
from flwr.common.logger import log
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from feddebug.utils import create_transform
from torchvision import transforms
from logging import INFO, WARNING


class ClientsAndServerDatasets:
    """Prepare the clients and server datasets for federated learning."""
    
    def __init__(self, cfg):
        self.cfg = cfg            
        self.client_id_to_loader = {}
        self.server_testloader = None
        self.clients_and_server_raw_data = None

        self._set_distribution_partitioner()
        self._load_datasets()
        self._introduce_label_noise()
    
    def _set_distribution_partitioner(self):
        """Set the data distribution partitioner based on configuration."""
        dist_type = self.cfg.data_dist.dist_type       
        if dist_type == 'iid':
            self.data_dist_partitioner_func = self._iid_data_distribution
        elif dist_type == 'non_iid_dirichlet':
            self.data_dist_partitioner_func = self._dirichlet_data_distribution
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")

    def _dirichlet_data_distribution(self, target_label_col: str = "label"):
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

    def _iid_data_distribution(self, target_label_col="label"):
        """Partition data using IID distribution."""
        partitioner = IidPartitioner(num_partitions=self.cfg.num_clients)
        return self._partition_helper(partitioner)
    
    def _partition_helper(self, partitioner):
        """Helper function to partition data."""
        fds = FederatedDataset(dataset=self.cfg.dataset.name, partitioners={"train": partitioner})
        server_data = fds.load_split("test")
        client2data = {f"{cid}": fds.load_partition(cid) for cid in range(self.cfg.num_clients)}
        return {'client2data': client2data, 'server_data': server_data}

    def _load_datasets(self):
        """Load and partition the datasets based on the partitioner."""
        self._validate_dataset()
        self.clients_and_server_raw_data = self.data_dist_partitioner_func(self.cfg)

        self._create_client_dataloaders()
        self.server_testloader = self._create_dataloader(self.clients_and_server_raw_data['server_data'], batch_size=512, shuffle=False)

    def _validate_dataset(self):
        """Validates if the selected dataset is supported."""
        supported_datasets = {"cifar10", "mnist"}  # Extend as needed
        if self.cfg.dataset.name.lower() not in supported_datasets:
            raise ValueError(f"Dataset '{self.cfg.dataset.name}' not supported.")
    
    def _create_client_dataloaders(self):
        """Create DataLoaders for each client."""
        self.client_id_to_loader = {
            client_id: self._create_dataloader(client_data, batch_size=self.cfg.client.batch_size)
            for client_id, client_data in self.clients_and_server_raw_data['client2data'].items()
            if client_id not in self.cfg.faulty_clients_ids
        }

    def _create_dataloader(self, ds, batch_size=64, shuffle=True):
        """Create a DataLoader with applied transformations."""
        def apply_transforms(batch):
            batch["image"] = [transform(img) for img in batch["image"]]
            return batch

        temp_transform = create_transform()

        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            temp_transform,
        ])
        transformed_dataset = ds.with_transform(apply_transforms)
        dataloader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader

    def _introduce_label_noise(self):
        """Introduce label noise to specified faulty clients."""
        faulty_client_ids = self.cfg.faulty_clients_ids
        noise_rate = self.cfg.noise_rate
        num_classes = self.cfg.dataset.num_classes
        client2data = self.clients_and_server_raw_data['client2data']

        for client_id in faulty_client_ids:
            client_ds = client2data[client_id]
            noisy_dataset = self._add_noise_in_data(client_ds, noise_rate, num_classes)
            self.client_id_to_loader[client_id] = self._create_dataloader(noisy_dataset, batch_size=self.cfg.client.batch_size)

        log(INFO, f"** All Malicious Clients are: {faulty_client_ids} **")

    def _add_noise_in_data(self, ds, noise_rate, num_classes):
        """Introduce label noise by flipping labels based on the noise rate."""
        def flip_labels(batch):
            labels = np.array(batch['label'])
            flip_mask = np.random.rand(len(labels)) < noise_rate
            indices_to_flip = np.where(flip_mask)[0]

            if len(indices_to_flip) > 0:
                new_labels = labels[indices_to_flip].copy()
                for idx in indices_to_flip:
                    current_label = new_labels[idx]
                    possible_labels = list(range(num_classes))
                    possible_labels.remove(current_label)
                    new_labels[idx] = random.choice(possible_labels)
                labels[indices_to_flip] = new_labels
                batch['label'] = labels
            return batch

        noisy_dataset = ds.map(flip_labels, batched=True, batch_size=256, num_proc=8).with_format("torch")
        return noisy_dataset

    def get_data(self):
        """Get the prepared client and server DataLoaders."""
        return {
            "server_testdata": self.server_testloader,
            "client2data": self.client_id_to_loader,
        }
