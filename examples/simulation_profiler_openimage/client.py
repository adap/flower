from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from dataset_utils import get_dataset
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, RandomSampler
from utils import model_to_arrays, set_params, test, train

from flwr.client import NumPyClient
from flwr.common.typing import Config, Metrics, NDArrays, Scalar

from dataloaders.openimage import OpenImage

from torchvision.models import shufflenet_v2_x2_0


class FlowerClient(NumPyClient):
    def __init__(self, *, cid: str, dataset: Dataset):
        self.cid = cid
        self.dataset = dataset

    def get_parameters(self, config: Config, model: Optional[Module] = None):
        if model is None:
            model = shufflenet_v2_x2_0(num_classes=596)

        return model_to_arrays(model)

    def fit(self, parameters, config):
        # Working with steps or local epochs
        replacement = True if "local_steps" in config.keys() else False
        local_steps = (
            int(config["local_steps"]) if "local_steps" in config.keys() else 0
        )
        num_samples = (
            local_steps * int(config["batch_size"])
            if local_steps != 0  # if local_steps not set by the server
            else len(self.dataset)
        )
        sampler = RandomSampler(
            self.dataset, replacement=replacement, num_samples=num_samples
        )
        self.dataset.load_client(cid=self.cid)
        dataloader = DataLoader(
            self.dataset, sampler=sampler, batch_size=int(config["batch_size"])
        )

        # Get assigned device from
        device = torch.device(f"cuda:{config['gpu_id']}")

        # Send model to device
        model = set_params(shufflenet_v2_x2_0(num_classes=596), parameters)
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
        dataset = get_dataset(Path(self.fed_dir).absolute(), str(config["cid"]), "val")
        dataloader = DataLoader(dataset, batch_size=int(config["batch_size"]))

        # Get assigned device from
        device = torch.device(f"cuda:{config['gpu_id']}")

        # Set parameters
        model = set_params(shufflenet_v2_x2_0(num_classes=596), parameters)
        model.to(device)

        # Evaluate
        loss, accuracy = test(model, dataloader, device=device)
        metrics: Metrics = {"accuracy": accuracy}

        # Return statistics
        return loss, len(dataset), metrics
