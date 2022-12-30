from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from dataset_utils import get_dataset
from torch.nn import Module
from torch.utils.data import DataLoader, RandomSampler
from utils import Net, model_to_arrays, set_params, test, train

from flwr.client import NumPyClient
from flwr.common.typing import Config, Metrics, NDArrays, Scalar


class FlowerClient(NumPyClient):
    def __init__(self, cid: str, fed_dir_data: str):
        self.cid = cid
        self.fed_dir = Path(fed_dir_data)
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}

    def get_parameters(self, config: Config, model: Optional[Module] = None):
        if model is None:
            model = Net()

        return model_to_arrays(model)

    def fit(self, parameters, config):

        # Load train data for this client
        dataset = get_dataset(Path(self.fed_dir).absolute(), self.cid, "train")

        replacement = True if "local_steps" in config.keys() else False
        local_steps = (
            int(config["local_steps"]) if "local_steps" in config.keys() else 0
        )
        num_samples = (
            local_steps * int(config["batch_size"])
            if local_steps != 0  # local_steps not set by the server
            else len(dataset)
        )
        sampler = RandomSampler(
            dataset, replacement=replacement, num_samples=num_samples
        )
        dataloader = DataLoader(
            dataset, sampler=sampler, batch_size=int(config["batch_size"])
        )

        # Get assigned device from
        device = torch.device(f"cuda:{config['gpu_id']}")

        # Send model to device
        model = set_params(Net(), parameters)
        model.to(device)

        # Train
        loss = train(
            model,
            dataloader,
            epochs=int(config["epochs"]),
            device=device,
        )
        metrics: Metrics = {"loss": loss, "accuracy": loss}

        # Return local model and statistics
        return model_to_arrays(model), len(dataset), metrics

    def evaluate(self, parameters: NDArrays, config: Config):
        # Load test data for this client
        dataset = get_dataset(Path(self.fed_dir).absolute(), self.cid, "val")
        dataloader = DataLoader(dataset, batch_size=int(config["batch_size"]))

        # Get assigned device from
        device = torch.device(f"cuda:{config['gpu_id']}")

        # Set parameters
        model = set_params(Net(), parameters)
        model.to(device)

        # Evaluate
        loss, accuracy = test(model, dataloader, device=device)
        metrics: Metrics = {"accuracy": accuracy}

        # Return statistics
        return loss, len(dataset), metrics
