"""FedAvg and FedNB clients."""

from collections import OrderedDict
from typing import Callable, Dict, List, Tuple
from pathlib import Path
import pickle

import flwr as fl
import torch
from flwr.common.typing import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from fedbn.models import CNNModel, test, train


class FlowerClient(fl.client.NumPyClient):
    """"""

    def __init__(
        self,
        model: CNNModel,
        trainloader: DataLoader,
        testloader: DataLoader,
        dataset_name: str,
        **kwargs,
    ) -> None:
        self.trainloader = trainloader
        self.testloader = testloader
        self.dataset_name = dataset_name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

    def get_parameters(self, config) -> NDArrays:
        """Return model parameters as a list of NumPy ndarrays w or w/o using BN
        layers.
        """
        # self.model.train() # TODO: is this needed ? check
        # Return all model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from a list of NumPy ndarrays Exclude the bn layer if
        available.
        """
        # self.model.train() # TODO: is this needed ? check
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Set model parameters, train model, return updated model parameters."""
        self.set_parameters(parameters)

        # evaluate the state of the global model on the train set; the loss returned
        # is what's reported in Fig3 in the FedBN paper (what this baseline focuses in reproducing)
        pre_train_loss, pre_train_acc = test(self.model, self.trainloader, device=self.device)

        # train model on local dataset
        loss, acc = train(
            self.model,
            self.trainloader,
            epochs=1,
            device=self.device,
        )

        # construct metrics to return to server
        round = config["round"]
        metrics = {
            "dataset_name": self.dataset_name,
            "round": round,
            "accuracy": acc,
            "loss": loss,
            "pre_train_loss": pre_train_loss,
            "pre_train_acc": pre_train_acc,
        }
        # print(f"Fit ({self.dataset_name}): {loss = } | {acc = }| num_samples: {len(self.trainloader.dataset)}" )
        return (
            self.get_parameters({}),
            len(self.trainloader.dataset),
            metrics,
        )

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Set model parameters, evaluate model on local test dataset, return result."""
        self.set_parameters(parameters)

        loss, accuracy = test(self.model, self.testloader, device=self.device)
        # print(f"Evaluate ({self.dataset_name}): {loss = } | {accuracy = } | num_samples: {len(self.testloader.dataset)}")
        return (
            float(loss),
            len(self.testloader.dataset),
            {"loss": loss, "accuracy": accuracy, "dataset_name": self.dataset_name},
        )


class FedBNFlowerClient(FlowerClient):

    def __init__(self, bn_state_dir: Path, client_id: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bn_state_dir = bn_state_dir
        self.bn_state_pkl = bn_state_dir/f"client_{client_id}.pkl"

    def _save_bn_statedict(self):

        bn_state = {name: val.cpu().numpy()
            for name, val in self.model.state_dict().items()
            if "bn" in name}
            
        with open(self.bn_state_pkl, "wb") as handle:
            pickle.dump(bn_state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_bn_statedict(self):
        with open(self.bn_state_pkl, "rb") as handle:
            data = pickle.load(handle)
        bn_stae_dict = {k: torch.tensor(v) for k, v in data.items()}
        return bn_stae_dict

    def get_parameters(self, config) -> NDArrays:
        """Return model parameters as a list of NumPy ndarrays w or w/o using BN
        layers.
        """
        # first update bn_state_dir
        self._save_bn_statedict()
        # self.model.train() # TODO: is this needed ? check
        # Excluding parameters of BN layers when using FedBN
        return [
            val.cpu().numpy()
            for name, val in self.model.state_dict().items()
            if "bn" not in name
        ]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from a list of NumPy ndarrays Exclude the bn layer if
        available.
        """
        # self.model.train() # TODO: is this needed ? check
        keys = [k for k in self.model.state_dict().keys() if "bn" not in k]
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=False)

        # now also load from bn_state_dir
        if self.bn_state_pkl.exists(): # it won't exist in the first round
            bn_state_dict = self._load_bn_statedict()
            self.model.load_state_dict(bn_state_dict, strict=False)


def gen_client_fn(
    client_data: List[Tuple[DataLoader,DataLoader,int]],
    client_cfg: DictConfig,
    model_cfg: DictConfig,
    bn_state_dir: Path,
) -> Callable[[str], FlowerClient]:
    """"""

    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""
        # Instantiate model
        net = instantiate(model_cfg)

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader, valloader, dataset_name = client_data[int(cid)]
        return instantiate(
            client_cfg, model=net, trainloader=trainloader, testloader=valloader, dataset_name=dataset_name,
            bn_state_dir=bn_state_dir, client_id=int(cid),
        )

    return client_fn
