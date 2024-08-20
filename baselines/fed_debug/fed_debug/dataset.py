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
from logging import INFO, WARNING

import torch
from flwr.common.logger import log

from fed_debug.dataset_preparation import (
    clients_data_distribution,
    prepare_iid_dataset,
    prepare_niid_dataset,
    train_test_transforms_factory,
)


class NoisyDataset(torch.utils.data.Dataset):
    """Dataset with noisy labels."""

    def __init__(self, dataset, num_classes, noise_rate):
        possible_noise_rates = [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1,
        ]
        assert (
            noise_rate in possible_noise_rates
        ), f"Noise rate must be in \
            /{possible_noise_rates} but got {noise_rate}"
        self.dataset = dataset
        self.num_classes = num_classes
        self.class_ids = random.sample(
            range(num_classes), int(noise_rate * num_classes)
        )

    def __len__(self):
        """Return the size of total dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Return the item at the given index."""
        x, y = self.dataset[idx]
        if y in self.class_ids:
            # random.seed(idx)
            y_hat = random.randint(0, self.num_classes - 1)
            if y_hat != y:
                y = y_hat
            else:
                y = (y + 1) % self.num_classes
        return x, y


def _add_noise_in_data(client_data, noise_rate, num_classes):
    return NoisyDataset(client_data, num_classes=num_classes, noise_rate=noise_rate)


def _initialize_image_dataset(cfg, fetch_only_test_data):
    """Initialize and return the image dataset."""
    target_label_col = "label"
    dataset_dict = clients_data_distribution(
        cfg, target_label_col, fetch_only_test_data
    )
    transforms = train_test_transforms_factory(cfg=cfg)
    dataset_dict["client2data"] = {
        k: v.map(
            transforms["train"], batched=True, batch_size=256, num_proc=8
        ).with_format("torch")
        for k, v in dataset_dict["client2data"].items()
    }
    dataset_dict["server_data"] = (
        dataset_dict["server_data"]
        .map(transforms["test"], batched=True, batch_size=256, num_proc=8)
        .with_format("torch")
    )
    return dataset_dict


def _load_datasets(cfg, fetch_only_test_data=False):
    """Load the dataset and return the dataload."""
    if cfg.dname in ["cifar10", "mnist", "flwrlabs/femnist"]:
        return _initialize_image_dataset(cfg, fetch_only_test_data)
    else:
        raise NotImplementedError(f"Dataset {cfg.dname} not implemented.")


def load_central_server_test_data(cfg):
    """Load the central server test data."""
    d_obj = ClientsAndServerDatasetsPrep(cfg).get_clients_server_data()
    return d_obj["server_testdata"]


class ClientsAndServerDatasetsPrep:
    """Prepare the clients and server datasets."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._setup()
        self._make_faulty_clients()
        self.client2data = {}
        self.server_testdata = None
        self.client2class = {}

    def _make_faulty_clients(self):
        """Make clients faulty."""
        for cid in self.cfg.faulty_clients_ids:
            self.client2data[cid] = _add_noise_in_data(
                client_data=self.client2data[cid],
                # label_col="label",
                noise_rate=self.cfg.noise_rate,
                num_classes=self.cfg.dataset.num_classes,
            )
            log(WARNING, f"Client {cid} is made noisy \n  ")
            self.client2class[cid] = "noisy"

    def _setup_hugging(self):
        dataset_dict = _load_datasets(self.cfg.data_dist)
        self.client2data = dataset_dict["client2data"]

        self.server_testdata = dataset_dict["server_data"]
        self.client2class = dataset_dict["client2class"]
        print(f"client2class: {self.client2class}")

        log(INFO, f"> client2class {self.client2class}")
        if len(self.client2data) < self.cfg.data_dist.num_clients:
            log(
                WARNING,
                f"orignal number of clients {self.cfg.data_dist.num_clients} "
                f"reduced to {len(self.client2data)}",
            )
            self.cfg.data_dist.num_clients = len(self.client2data)

        data_per_client = [len(dl) for dl in self.client2data.values()]
        log(INFO, f"Data per client in experiment {data_per_client}")
        min_data = min(len(dl) for dl in self.client2data.values())
        log(INFO, f"Min data on a client: {min_data}")

    def _setup_original_fed_debug(self):
        self.client2class = {}
        self.server_testdata = None
        self.client2data = None

        if self.cfg.data_dist.dist_type == "iid":
            self.client2data, self.server_testdata, _ = prepare_iid_dataset(
                dname=self.cfg.dataset.name,
                dataset_dir=self.cfg.storage.dir,
                num_clients=self.cfg.num_clients,
            )

        elif self.cfg.data_dist.dist_type == "niid":
            self.client2data, self.server_testdata, _ = prepare_niid_dataset(
                dname=self.cfg.dataset.name,
                dataset_dir=self.cfg.storage.dir,
                num_clients=self.cfg.num_clients,
            )

    def _setup(self):
        self._setup_original_fed_debug()
        # self._setup_hugging()

    def get_clients_server_data(self):
        """Return the clients and server data for simulation."""
        return {
            "server_testdata": self.server_testdata,
            "client2class": self.client2class,
            "client2data": self.client2data,
        }
