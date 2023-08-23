# pylint: disable=too-many-arguments
"""Defines the MNIST Flower Client and a function to instantiate it."""


from collections import OrderedDict
from typing import Callable, Dict, Tuple

from pathlib import Path
import pickle
import flwr as fl
import torch
from flwr.common.typing import NDArrays, Scalar
from torch.utils.data import DataLoader

from FedPer_new.model import DecoupledModel, train, test
from FedPer_new.utils import MEAN, STD
from FedPer_new.dataset_preparation import DATASETS
from typing import List

from hydra.utils import instantiate

from torchvision import transforms
from torch.utils.data import Subset

PROJECT_DIR = Path(__file__).parent.parent.absolute()


class FlowerClient(fl.client.NumPyClient):
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        net: DecoupledModel,
        trainloader: Subset,
        valloader: Subset,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
    ):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        self.client_id : int = None

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Returns the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Changes the parameters of the model using the given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {
                k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0])
                for k, v in params_dict
            }
        )
        self.net.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implements distributed fit function for a given client."""
        self.set_parameters(parameters)
        train(
            self.net,
            self.trainloader,
            self.device,
            epochs=self.num_epochs,
            learning_rate=self.learning_rate,
        )
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implements distributed evaluation for a given client."""
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

def gen_client_fn(
    config: dict
) -> Tuple[Callable[[str], FlowerClient], DataLoader]:
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

    def client_fn(cid: str) -> FlowerClient:
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
        return FlowerClient(
            net, 
            trainloader, 
            testloader, 
            config.server_device, 
            config.num_epochs, 
            config.learning_rate
        )

    return client_fn