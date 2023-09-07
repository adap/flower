"""Client implementation - can call FedPer and FedAvg clients."""
import copy
import pickle
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import flwr as fl
import numpy as np
import torch
from flwr.common import Scalar
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

from FedPer.constants import DEFAULT_FT_EP, MEAN, STD, Algorithms
from FedPer.dataset_preparation import DATASETS
from FedPer.implemented_models.cnn_model import CNNModelManager
from FedPer.implemented_models.mobile_model import MobileNetModelManager
from FedPer.implemented_models.resnet_model import ResNetModelManager

PROJECT_DIR = Path(__file__).parent.parent.absolute()


class BaseClient(fl.client.NumPyClient):
    """Implementation of Federated Averaging (FedAvg) Client."""

    def __init__(
        self,
        trainloader: DataLoader,
        testloader: DataLoader,
        config: Dict[str, Any],
        client_id: int,
        model_manager_class: Union[
            Type[CNNModelManager], Type[MobileNetModelManager], Type[ResNetModelManager]
        ],
        has_fixed_head: bool = False,
        client_state_save_path: str = None,
    ):
        """Initialize client attributes.

        Args:
            config: dictionary containing the client configurations.
            client_id: id of the client.
            model_manager_class: class to be used as the model manager.
            has_fixed_head: whether a fixed head should be used or not.
        """
        super().__init__()

        self.train_id = 1
        self.test_id = 1
        self.config = config
        self.client_id = client_id
        try:
            self.client_state_save_path = (
                client_state_save_path + f"/client_{self.client_id}"
            )
        except TypeError:
            self.client_state_save_path = None
        self.hist: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.model_manager = model_manager_class(
            client_id=self.client_id,
            config=config,
            has_fixed_head=has_fixed_head,
            trainloader=trainloader,
            testloader=testloader,
            client_save_path=self.client_state_save_path,
        )

    def get_parameters(self) -> List[np.ndarray]:
        """Return the current local model parameters."""
        return self.model_manager.model.get_parameters()

    def set_parameters(
        self, parameters: List[np.ndarray], evaluate: bool = False
    ) -> None:
        """Set the local model parameters to the received parameters.

        Args:
            parameters: parameters to set the model to.
        """
        model_keys = [
            k
            for k in self.model_manager.model.state_dict().keys()
            if k.startswith("_body") or k.startswith("_head")
        ]
        params_dict = zip(model_keys, parameters)

        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

        self.model_manager.model.set_parameters(state_dict)

    def perform_train(
        self, tag: str = None
    ) -> Dict[str, Union[List[Dict[str, float]], int, float]]:
        """Perform local training to the whole model.

        Args:
            tag: str of the form <Algorithm>_<model_train_part>.
                <Algorithm> - indicates the federated algorithm that is being performed\
                    (FedAvg, FedPer).
                <model_train_part> - indicates the part of the model that is
                    being trained (full, body, head).
                    This tag can be ignored if no difference in train behaviour is
                    desired between federated algortihms.

        Returns
        -------
            Dict with the train metrics.
        """
        epochs = self.config.get("epochs", {"full": 4})
        print("Epochs: ", epochs)
        print("Tag: ", tag)

        self.model_manager.model.enable_body()
        self.model_manager.model.enable_head()

        return self.model_manager.train(
            train_id=self.train_id,
            epochs=epochs.get("full", 4),
            tag="FedAvg_full" if tag is None else tag,
        )

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Union[
        Tuple[List[np.ndarray], int, Dict[str, Scalar]], Tuple[List[np.ndarray], int]
    ]:
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
        self.hist[str(self.train_id)] = {
            **self.hist[str(self.train_id)],
            "trn": train_results,
        }
        print("<------- TRAIN RESULTS -------> :", train_results)

        self.train_id += 1

        return self.get_parameters(), self.model_manager.train_dataset_size(), {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Union[
        Tuple[float, int, Dict[str, Scalar]],
        Tuple[int, float, float],
        Tuple[int, float, float, Dict[str, Scalar]],
    ]:
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

        if self.config.get("fine-tuning", False):
            # Save the model parameters before fine-tuning
            model_state_dict = copy.deepcopy(self.model_manager.model.state_dict())

            # Fine-tune the model before testing
            epochs = self.config.get("epochs", {"fine-tuning": DEFAULT_FT_EP})
            ft_trn_results = self.model_manager.train(
                train_id=self.test_id,
                epochs=epochs.get("fine-tuning", DEFAULT_FT_EP),
                fine_tuning=True,
                tag=f"{self.config.get('algorithm', 'FedAvg')}_full",
            )

        # Test the model
        tst_results = self.model_manager.test(test_id=self.test_id)
        print("<------- TEST RESULTS -------> :", tst_results)

        # Update test history
        self.hist[str(self.test_id)] = {
            **self.hist[str(self.test_id)],
            "tst": tst_results,
        }

        if self.config.get("fine-tuning", False):
            # Set the model parameters as they were before fine-tuning
            self.model_manager.model.set_parameters(model_state_dict)

            # Update the history with the ft_trn results
            self.hist[str(self.test_id)]["trn"] = {
                **(self.hist[str(self.test_id)].get("trn", {})),
                **ft_trn_results,
            }
        self.test_id += 1

        return (
            tst_results.get("loss", 0.0),
            self.model_manager.test_dataset_size(),
            {k: v for k, v in tst_results.items() if not isinstance(v, (dict, list))},
        )


class FedPerClient(BaseClient):
    """Implementation of Federated Personalization (FedPer) Client."""

    def get_parameters(self) -> List[np.ndarray]:
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
            print("Setting head parameters to global head parameters.")
            model_keys.extend(
                [
                    k
                    for k in self.model_manager.model.state_dict().keys()
                    if k.startswith("_head")
                ]
            )

        params_dict = zip(model_keys, parameters)

        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

        self.model_manager.model.set_parameters(state_dict)

    def perform_train(
        self, tag: str = None
    ) -> Dict[str, Union[List[Dict[str, float]], int, float]]:
        """Perform local training to the whole model.

        Args:
            tag: str of the form <Algorithm>_<model_train_part>.
                <Algorithm> - indicates the federated algorithm that is being performed\
                    (FedAvg, FedPer).
                <model_train_part> - indicates the part of the model that is
                being trained (full, body, head). This tag can be ignored if no
                difference in train behaviour is desired between federated algortihms.

        Returns
        -------
            Dict with the train metrics.
        """
        return super().perform_train(
            tag=f"{Algorithms.FEDPER.value}_full" if tag is None else tag
        )


def get_client_fn_simulation(
    config: DictConfig,
    client_state_save_path: str = None,
) -> Tuple[Callable[[str], BaseClient], DataLoader]:
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
    assert config.model.name.lower() in [
        "cnn",
        "mobile",
        "resnet",
    ], f"Model {config.model.name} not implemented"
    # load dataset and clients' data indices
    if config.dataset.name.lower() in ["cifar10", "cifar100"]:
        try:
            partition_path = (
                PROJECT_DIR / "datasets" / config.dataset.name / "partition.pkl"
            )
            print(f"Loading partition from {partition_path}")
            with open(partition_path, "rb") as f:
                partition = pickle.load(f)
            data_indices: List[List[int]] = partition["data_indices"]
        except FileNotFoundError as e:
            print(f"Partition not found at {partition_path}")
            raise e

        # - you can define your own data transformation strategy here -
        general_data_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    MEAN[config.dataset.name], STD[config.dataset.name]
                ),
            ]
        )
        general_target_transform = transforms.Compose([])
        train_data_transform = transforms.Compose([])
        train_target_transform = transforms.Compose([])
        # ------------------------------------------------------------

    def client_fn(cid: str) -> BaseClient:
        """Create a Flower client representing a single organization."""
        cid = int(cid)
        if config.dataset.name.lower() == "flickr":
            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ]
            )
            data_path = (
                PROJECT_DIR / "datasets" / config.dataset.name / "tmp" / f"client_{cid}"
            )
            dataset = ImageFolder(root=data_path, transform=transform)
            trainset, testset = random_split(
                dataset,
                [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)],
            )
        else:
            dataset = DATASETS[config.dataset.name](
                root=PROJECT_DIR / "datasets" / config.dataset.name,
                config=config.dataset,
                general_data_transform=general_data_transform,
                general_target_transform=general_target_transform,
                train_data_transform=train_data_transform,
                train_target_transform=train_target_transform,
            )
            trainset = Subset(dataset, indices=[])
            testset = Subset(dataset, indices=[])
            trainset.indices = data_indices[cid]["train"]
            testset.indices = data_indices[cid]["test"]

        # Create the train loader
        trainloader = DataLoader(trainset, config.batch_size, shuffle=True)
        # Create the test loader
        testloader = DataLoader(testset, config.batch_size)

        if config.model.name.lower() == "cnn":
            manager = CNNModelManager
        elif config.model.name.lower() == "mobile":
            manager = MobileNetModelManager
        elif config.model.name.lower() == "resnet":
            manager = ResNetModelManager
        else:
            raise NotImplementedError("Model not implemented, check name.")
        if client_state_save_path is not None:
            return FedPerClient(
                trainloader=trainloader,
                testloader=testloader,
                client_id=cid,
                config=config.model,
                model_manager_class=manager,
                client_state_save_path=client_state_save_path,
            )
        else:
            return BaseClient(
                trainloader=trainloader,
                testloader=testloader,
                client_id=cid,
                config=config.model,
                model_manager_class=manager,
                client_state_save_path=client_state_save_path,
            )

    return client_fn
