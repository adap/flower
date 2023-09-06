
import torch
import numpy as np

from typing import Dict, List, Union, Tuple, Callable
from omegaconf import DictConfig
from collections import OrderedDict
from torch.utils.data import DataLoader
from FedPer.models.cnn_model import CNNModelManager
from FedPer.models.mobile_model import MobileNetModelManager
from FedPer.models.resnet_model import ResNetModelManager
from FedPer.utils.constants import Algorithms
from FedPer.utils.base_client import BaseClient

from pathlib import Path
import pickle
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
from FedPer.dataset_preparation import DATASETS

from tqdm import tqdm
from FedPer.utils.utils_file import MEAN, STD

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()

class FedPerClient(BaseClient):
    """Implementation of Federated Learning with Personalization Layers (FedPer) Client."""

    def get_parameters(self) -> List[np.ndarray]:
        """Return the current local head parameters."""
        return [val.cpu().numpy() for _, val in self.model_manager.model.body.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray], evaluate=False) -> None:
        """
        Set the local body parameters to the received parameters.
        In the first train round the head parameters are also set to the global head parameters,
        to ensure every client head is initialized equally.

        Args:
            parameters: parameters to set the body to.
        """
        model_keys = [k for k in self.model_manager.model.state_dict().keys() if k.startswith("_body")]

        if not evaluate:
            # Only update client's local head if it hasn't trained yet
            print("Setting head parameters to global head parameters.")
            model_keys.extend([k for k in self.model_manager.model.state_dict().keys() if k.startswith("_head")])

        params_dict = zip(model_keys, parameters)

        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

        self.model_manager.model.set_parameters(state_dict)

    def perform_train(self, tag: str = None) -> Dict[str, Union[List[Dict[str, float]], int, float]]:
        """
        Perform local training to the whole model.

        Args:
            tag: str of the form <Algorithm>_<model_train_part>.
                <Algorithm> - indicates the federated algorithm that is being performed\
                              (FedAvg, FedPer, FedRep, FedBABU or FedHybridAvgLGDual).
                              In the case of FedHybridAvgLGDual the tag also includes which part of the algorithm\
                                is being performed, either FedHybridAvgLGDual_FedAvg or FedHybridAvgLGDual_LG-FedAvg.
                <model_train_part> - indicates the part of the model that is being trained (full, body, head).
                This tag can be ignored if no difference in train behaviour is desired between federated algortihms.
        Returns:
            Dict with the train metrics.
        """

        return super().perform_train(tag=f"{Algorithms.FEDPER.value}_full" if tag is None else tag)

def get_fedper_client_fn(
    cfg: DictConfig,
    client_state_save_path: str,
) -> Tuple[
    Callable[[str], FedPerClient], DataLoader
]:  # pylint: disable=too-many-arguments
    """Generates the client function that creates the Flower Clients.

    Parameters
    ----------
    cfg : DictConfig
        The model configuration.

    client_state_save_path : str
        The path to save the client state.
    Returns
    -------
    Tuple[Callable[[str], FlowerClient], DataLoader]
        A tuple containing the client function that creates Flower Clients and
        the DataLoader that will be used for testing
    """

    assert cfg.model.name.lower() in ['cnn', 'mobile', 'resnet']
    assert client_state_save_path is not None, "Please provide a path to save the client state."
    # load dataset and clients' data indices
    if cfg.dataset.name in ["cifar10", "cifar100"]:
        try:
            partition_path = PROJECT_DIR / "datasets" / cfg.dataset.name / "partition.pkl"
            print(f"Loading partition from {partition_path}")
            with open(partition_path, "rb") as f:
                partition = pickle.load(f)
        except:
            raise FileNotFoundError(f"Please partition {cfg.dataset.name} first.")

        data_indices: List[List[int]] = partition["data_indices"]

        # --------- you can define your own data transformation strategy here ------------
        #general_data_transform = transforms.Compose(
        #    [transforms.Normalize(MEAN[cfg.dataset.name], STD[cfg.dataset.name])]
        #)

        general_data_transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            # transforms.ToTensor(),
            transforms.Normalize(MEAN[cfg.dataset.name], STD[cfg.dataset.name])
        ])

        general_target_transform = transforms.Compose([])
        train_data_transform = transforms.Compose([])
        train_target_transform = transforms.Compose([])
        # --------------------------------------------------------------------------------

        dataset = DATASETS[cfg.dataset.name](
            root=PROJECT_DIR / "datasets" / cfg.dataset.name,
            config=cfg.dataset,
            general_data_transform=general_data_transform,
            general_target_transform=general_target_transform,
            train_data_transform=train_data_transform,
            train_target_transform=train_target_transform,
        )

        trainset: Subset = Subset(dataset, indices=[])
        testset: Subset = Subset(dataset, indices=[])

    def client_fn(cid: str) -> FedPerClient:
        """Create a Flower client representing a single organization."""

        cid = int(cid)
        if cfg.dataset.name.lower() in ['cifar10', 'cifar100']:
            trainset.indices = data_indices[cid]["train"]
            testset.indices = data_indices[cid]["test"]
        else:
            transform = transforms.Compose([
                # resize
                transforms.Resize((224, 224)),
                # convert to tensor
                transforms.ToTensor(),
                # normalize
                # transforms.Normalize(MEAN[cfg.dataset.name], STD[cfg.dataset.name])
            ])
            from torchvision.datasets import ImageFolder
            from torchvision.transforms import ToTensor
            from torch.utils.data import DataLoader, random_split
            data_path = PROJECT_DIR / "datasets" / cfg.dataset.name / "tmp" / f"client_{cid}"
            data = ImageFolder(root=data_path, transform=transform)
            trainset, testset = random_split(data, [int(len(data)*0.8), len(data)-int(len(data)*0.8)])

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = DataLoader(trainset, cfg.batch_size)
        testloader = DataLoader(testset, cfg.batch_size)

        if cfg.model.name.lower() == 'cnn':
            manager = CNNModelManager
        elif cfg.model.name.lower() == 'mobile':
            manager = MobileNetModelManager
        elif cfg.model.name.lower() == 'resnet':
            manager = ResNetModelManager
        else:
            raise NotImplementedError('Model not implemented, check name.')

        return FedPerClient(
            trainloader=trainloader,
            testloader=testloader,
            client_id=cid,
            config=cfg.model,
            model_manager_class=manager,
            client_state_save_path=client_state_save_path,
        )
    
    return client_fn