"""Client implementation - can call FedPer and FedAvg clients."""

import pickle
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import numpy as np
import torch
from flwr.client import Client, NumPyClient
from flwr.common import NDArrays, Scalar
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from fedrep.constants import MEAN, STD
from fedrep.dataset_preparation import call_dataset
from fedrep.implemented_models.cnn_cifar10 import CNNCifar10ModelManager
from fedrep.implemented_models.cnn_cifar100 import CNNCifar100ModelManager

PROJECT_DIR = Path(__file__).parent.parent.absolute()


class BaseClient(NumPyClient):
    """Implementation of Federated Averaging (FedAvg) Client."""

    # pylint: disable=R0913
    def __init__(
        self,
        client_id: int,
        trainloader: DataLoader,
        testloader: DataLoader,
        config: DictConfig,
        model_manager_class: Union[
            Type[CNNCifar10ModelManager], Type[CNNCifar100ModelManager]
        ],
        client_state_save_path: str = "",
    ):
        """Initialize client attributes.

        Args:
            client_id: The client ID.
            trainloader: Client train data loader.
            testloader: Client test data loader.
            config: dictionary containing the client configurations.
            model_manager_class: class to be used as the model manager.
            client_state_save_path: Path for saving model head parameters.
                (Just for FedRep). Defaults to "".
        """
        super().__init__()

        self.client_id = client_id
        self.client_state_save_path = (
            (client_state_save_path + f"/client_{self.client_id}")
            if client_state_save_path != ""
            else None
        )
        self.hist: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.num_epochs: int = config["num_epochs"]
        self.model_manager = model_manager_class(
            client_id=self.client_id,
            config=config,
            trainloader=trainloader,
            testloader=testloader,
            client_save_path=self.client_state_save_path,
            learning_rate=config["learning_rate"],
        )

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the current local model parameters."""
        return self.model_manager.model.get_parameters()

    def set_parameters(
        self, parameters: List[np.ndarray], evaluate: bool = False
    ) -> None:
        """Set the local model parameters to the received parameters.

        Args:
            parameters: parameters to set the model to.
        """
        _ = evaluate
        model_keys = [
            k
            for k in self.model_manager.model.state_dict().keys()
            if k.startswith("_body") or k.startswith("_head")
        ]
        params_dict = zip(model_keys, parameters)

        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

        self.model_manager.model.set_parameters(state_dict)

    def perform_train(self) -> Dict[str, Union[List[Dict[str, float]], int, float]]:
        """Perform local training to the whole model.

        Returns
        -------
            Dict with the train metrics.
        """
        self.model_manager.model.enable_body()
        self.model_manager.model.enable_head()

        return self.model_manager.train()

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Union[bool, bytes, float, int, str]]]:
        """Train the provided parameters using the locally held dataset.

        Args:
            parameters: The current (global) model parameters.
            config: configuration parameters for training sent by the server.

        Returns
        -------
            Tuple containing the locally updated model parameters, \
                the number of examples used for training and \
                the training metrics.
        """
        self.set_parameters(parameters)

        train_results = self.perform_train()

        # Update train history
        print("<------- TRAIN RESULTS -------> :", train_results)

        return self.get_parameters(config), self.model_manager.train_dataset_size(), {}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Union[bool, bytes, float, int, str]]]:
        """Evaluate the provided global parameters using the locally held dataset.

        Args:
            parameters: The current (global) model parameters.
            config: configuration parameters for training sent by the server.

        Returns
        -------
        Tuple containing the test loss, \
                the number of examples used for evaluation and \
                the evaluation metrics.
        """
        self.set_parameters(parameters, evaluate=True)

        # Test the model
        test_results = self.model_manager.test()
        print("<------- TEST RESULTS -------> :", test_results)

        return (
            test_results.get("loss", 0.0),
            self.model_manager.test_dataset_size(),
            {k: v for k, v in test_results.items() if not isinstance(v, (dict, list))},
        )


class FedRepClient(BaseClient):
    """Implementation of Federated Personalization (FedRep) Client."""

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the current local body parameters."""
        return [
            val.cpu().numpy()
            for _, val in self.model_manager.model.body.state_dict().items()
        ]

    def set_parameters(self, parameters: List[np.ndarray], evaluate=False) -> None:
        """Set the local body parameters to the received parameters.

        Args:
            parameters: parameters to set the body to.
            evaluate: whether the client is evaluating or not.
        """
        model_keys = [
            k
            for k in self.model_manager.model.state_dict().keys()
            if k.startswith("_body")
        ]

        if not evaluate:
            # Only update client's local head if it hasn't trained yet
            model_keys.extend(
                [
                    k
                    for k in self.model_manager.model.state_dict().keys()
                    if k.startswith("_head")
                ]
            )

        state_dict = OrderedDict(
            (k, torch.from_numpy(v)) for k, v in zip(model_keys, parameters)
        )

        self.model_manager.model.set_parameters(state_dict)


def get_client_fn_simulation(
    config: DictConfig, client_state_save_path: str = ""
) -> Callable[[str], Client]:
    """Generate the client function that creates the Flower Clients.

    Parameters
    ----------
    model : DictConfig
        The model configuration.
    cleint_state_save_path : str
        The path to save the client state.

    Returns
    -------
    Tuple[Callable[[str], FlowerClient], DataLoader]
        A tuple containing the client function that creates Flower Clients and
        the DataLoader that will be used for testing
    """
    assert config.model_name.lower() in [
        "cnncifar10",
        "cnncifar100",
    ], f"Model {config.model_name} not implemented"

    # load dataset (cifar10/cifar100) and clients' data indices
    try:
        partition_path = (
            PROJECT_DIR / "datasets" / config.dataset.name / "partition.pkl"
        )
        print(f"Loading partition from {partition_path}")
        with open(partition_path, "rb") as pickle_file:
            partition = pickle.load(pickle_file)
        data_indices: Dict[int, Dict[str, List[int]]] = partition["data_indices"]
    except FileNotFoundError as error:
        print(f"Partition not found at {partition_path}")
        raise error

    # - you can define your own data transformation strategy here -
    train_data_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            transforms.Normalize(MEAN[config.dataset.name], STD[config.dataset.name]),
        ]
    )
    test_data_transform = transforms.Compose(
        [
            # transforms.ToTensor(),
            transforms.Normalize(MEAN[config.dataset.name], STD[config.dataset.name])
        ]
    )
    dataset = call_dataset(
        dataset_name=config.dataset.name,
        root=PROJECT_DIR / "datasets" / config.dataset.name,
        train_data_transform=train_data_transform,
        test_data_transform=test_data_transform,
    )

    def client_fn(cid: str) -> Client:
        """Create a Flower client representing a single organization."""
        cid_use = int(cid)

        trainset = Subset(dataset, indices=data_indices[cid_use]["train"])
        testset = Subset(dataset, indices=data_indices[cid_use]["test"])

        # Create the train loader
        trainloader = DataLoader(trainset, config.batch_size, shuffle=True)
        # Create the test loader
        testloader = DataLoader(testset, config.batch_size)

        model_manager_class: Union[
            Type[CNNCifar10ModelManager], Type[CNNCifar100ModelManager]
        ]
        if config.model_name.lower() == "cnncifar10":
            model_manager_class = CNNCifar10ModelManager
        elif config.model_name.lower() == "cnncifar100":
            model_manager_class = CNNCifar100ModelManager
        else:
            raise NotImplementedError(
                f"Model {config.model_name} not implemented, check name."
            )

        if client_state_save_path != "":
            return FedRepClient(  # type: ignore[attr-defined]
                client_id=cid_use,
                trainloader=trainloader,
                testloader=testloader,
                config=config,
                model_manager_class=model_manager_class,
                client_state_save_path=client_state_save_path,
            ).to_client()
        return BaseClient(  # type: ignore[attr-defined]
            client_id=cid_use,
            trainloader=trainloader,
            testloader=testloader,
            config=config,
            model_manager_class=model_manager_class,
            client_state_save_path=client_state_save_path,
        ).to_client()

    return client_fn
