"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""

import flwr as fl
import torch
import numpy as np
import pickle
import torch.nn as nn

from typing import Callable, Dict, List, Tuple, Union
from pathlib import Path
from omegaconf import DictConfig
from collections import OrderedDict
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from flwr.common.typing import NDArrays, Scalar

from FedPer_new.model import DecoupledModel, train, test
from FedPer_new.utils import MEAN, STD
from FedPer_new.dataset_preparation import DATASETS

from FedPer_new.fedavg_client import FlowerClient as FedAvgFlowerClient

from hydra.utils import instantiate
from torchvision import transforms
from torch.utils.data import Subset

PROJECT_DIR = Path(__file__).parent.parent.absolute()

class FedPerClient(
    FedAvgFlowerClient
):  # pylint: disable=too-many-instance-attributes
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        net: torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
    ):  # pylint: disable=too-many-arguments
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.train_id = 1

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Returns the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.net.body.state_dict().items()]
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """
        Set the local body parameters to the received parameters.
        In the first train round the head parameters are also set to the global head parameters,
        to ensure every client head is initialized equally.

        Args:
            parameters: parameters to set the body to.
        """
        # Get model keys for body
        model_keys = [k for k in self.net.state_dict().keys() if k.startswith("body")]
        # print("Model keys: ", model_keys)
        if self.train_id == 1:
            # Only update client's local head if it hasn't trained yet
            model_keys.extend([k for k in self.net.state_dict().keys() if k.startswith("classifier")])
        # Zip model keys and parameters
        params_dict = zip(model_keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        # self.net.set_parameters(state_dict)
        self.net.load_state_dict(state_dict, strict=True)

    def fit(
        self, 
        parameters: NDArrays, 
        config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implements distributed fit function for a given client."""
        # Set parameters 
        self.set_parameters(parameters)
        # Epochs
        epochs = config["epochs"]

        # Train model
        train(
            net=self.net,
            trainloader=self.trainloader,
            device=self.device,
            epochs=epochs,
            learning_rate=self.learning_rate,
        )

        self.train_id += 1

        return self.get_parameters({}), len(self.trainloader), {}

    # super class evaluate function
    def evaluate(
        self, 
        parameters: NDArrays, 
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        super().evaluate(parameters, config)     
        

def gen_client_fn(
    config: dict
) -> Tuple[Callable[[str], FedPerClient], DataLoader]:
    """Generates the client function that creates the Flower Clients.

    Parameters
    ----------
    config : dict
        The configuration dictionary containing the parameters for the client

    Returns
    -------
    Tuple[Callable[[str], FlowerClient], DataLoader]
        A tuple containing the client function that creates Flower Clients and
        the DataLoader that will be used for testing
    """
    # load dataset and clients' data indices
    try:
        partition_path = PROJECT_DIR / "datasets" / config.dataset.name / "partition.pkl"
        print(f"Loading partition from {partition_path}")
        with open(partition_path, "rb") as f:
            partition = pickle.load(f)
    except:
        raise FileNotFoundError(f"Please partition {config.dataset.name} first.")

    data_indices: List[List[int]] = partition["data_indices"]

    # --------- you can define your own data transformation strategy here ------------
    general_data_transform = transforms.Compose(
        [transforms.Normalize(MEAN[config.dataset.name], STD[config.dataset.name])]
    )
    general_target_transform = transforms.Compose([])
    train_data_transform = transforms.Compose([])
    train_target_transform = transforms.Compose([])
    # --------------------------------------------------------------------------------

    dataset = DATASETS[config.dataset.name](
        root=PROJECT_DIR / "data" / config.dataset.name,
        config=config.dataset,
        general_data_transform=general_data_transform,
        general_target_transform=general_target_transform,
        train_data_transform=train_data_transform,
        train_target_transform=train_target_transform,
    )

    trainset: Subset = Subset(dataset, indices=[])
    testset: Subset = Subset(dataset, indices=[])
    global_testset: Subset = None
    if config['global_testset']:
        all_testdata_indices = []
        for indices in data_indices:
            all_testdata_indices.extend(indices["test"])
        global_testset = Subset(dataset, all_testdata_indices)

    # Get model
    model = instantiate(config.model)

    def client_fn(cid: str) -> FedPerClient:
        """Create a Flower client representing a single organization."""

        # Load model
        net = model.to(config.model.device)

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        cid = int(cid)
        trainset.indices = data_indices[cid]["train"]
        testset.indices = data_indices[cid]["test"]
        trainloader = DataLoader(trainset, config.batch_size)
        if config.global_testset:
            testloader = DataLoader(global_testset, config.batch_size)
        else:
            testloader = DataLoader(testset, config.batch_size)

        # Create a  single Flower client representing a single organization
        return FedPerClient(
            net, 
            trainloader, 
            testloader, 
            config.server_device, 
            config.num_epochs, 
            config.learning_rate
        )

    return client_fn