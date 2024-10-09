"""Client implementation - can call FedPep and FedAvg clients."""

from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Type, Union

import numpy as np
import torch
from flwr.client import Client, NumPyClient
from flwr.common import NDArrays, Scalar
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import PathologicalPartitioner
from flwr_datasets.preprocessor import Merger
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms

from fedrep.constants import MEAN, STD, Algorithm
from fedrep.models import CNNCifar10ModelManager, CNNCifar100ModelManager

PROJECT_DIR = Path(__file__).parent.parent.absolute()

FEDERATED_DATASET = None


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
        self.model_manager = model_manager_class(
            client_id=self.client_id,
            config=config,
            trainloader=trainloader,
            testloader=testloader,
            client_save_path=self.client_state_save_path,
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
            for val in self.model_manager.model.body.state_dict().values()
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


# pylint: disable=E1101, W0603
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

    # - you can define your own data transformation strategy here -
    # These transformations are from the official repo
    train_data_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN[config.dataset.name], STD[config.dataset.name]),
        ]
    )
    test_data_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN[config.dataset.name], STD[config.dataset.name]),
        ]
    )

    use_fine_label = False
    if config.dataset.name.lower() == "cifar100":
        use_fine_label = True

    partitioner = PathologicalPartitioner(
        num_partitions=config.num_clients,
        partition_by="fine_label" if use_fine_label else "label",
        num_classes_per_partition=config.dataset.num_classes,
        class_assignment_mode="random",
        shuffle=True,
        seed=config.dataset.seed,
    )

    global FEDERATED_DATASET
    if FEDERATED_DATASET is None:
        FEDERATED_DATASET = FederatedDataset(
            dataset=config.dataset.name.lower(),
            partitioners={"all": partitioner},
            preprocessor=Merger({"all": ("train", "test")}),
        )

    def apply_train_transforms(batch):
        """Apply transforms for train data to the partition from FederatedDataset."""
        batch["img"] = [train_data_transform(img) for img in batch["img"]]
        if use_fine_label:
            batch["label"] = batch["fine_label"]
        return batch

    def apply_test_transforms(batch):
        """Apply transforms for test data to the partition from FederatedDataset."""
        batch["img"] = [test_data_transform(img) for img in batch["img"]]
        if use_fine_label:
            batch["label"] = batch["fine_label"]
        return batch

    # pylint: disable=E1101
    def client_fn(cid: str) -> Client:
        """Create a Flower client representing a single organization."""
        cid_use = int(cid)

        partition = FEDERATED_DATASET.load_partition(cid_use, split="all")

        partition_train_test = partition.train_test_split(
            train_size=config.dataset.fraction, shuffle=True, seed=config.dataset.seed
        )

        trainset = partition_train_test["train"].with_transform(apply_train_transforms)
        testset = partition_train_test["test"].with_transform(apply_test_transforms)

        trainloader = DataLoader(trainset, config.batch_size, shuffle=True)
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

        if config.algorithm.lower() == Algorithm.FEDREP.value:
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
