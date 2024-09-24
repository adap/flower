"""FedAvg and FedNB clients."""

import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import flwr as fl
import torch
from flwr.common.typing import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from fedbn.models import CNNModel, test, train


class FlowerClient(fl.client.NumPyClient):
    """A standard FlowerClient.

    This base class is what plain FedAvg clients do.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        model: CNNModel,
        trainloader: DataLoader,
        testloader: DataLoader,
        dataset_name: str,
        l_r: float,
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        self.trainloader = trainloader
        self.testloader = testloader
        self.dataset_name = dataset_name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.l_r = l_r

    def get_parameters(self, config) -> NDArrays:
        """Return model parameters as a list of NumPy ndarrays w or w/o.

        using BNlayers.
        """
        # Return all model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from a list of NumPy ndarrays Exclude the BN.

        layer if available.
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Set model parameters, train model, return updated model parameters."""
        self.set_parameters(parameters)

        # Evaluate the state of the global model on the train set; the loss returned
        # is what's reported in Fig3 in the FedBN paper (what this baseline focuses
        # in reproducing)
        pre_train_loss, pre_train_acc = test(
            self.model, self.trainloader, device=self.device
        )

        # Train model on local dataset
        loss, acc = train(
            self.model,
            self.trainloader,
            epochs=1,
            l_r=self.l_r,
            device=self.device,
        )

        # Construct metrics to return to server
        fl_round = config["round"]
        metrics = {
            "dataset_name": self.dataset_name,
            "round": fl_round,
            "accuracy": acc,
            "loss": loss,
            "pre_train_loss": pre_train_loss,
            "pre_train_acc": pre_train_acc,
        }

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
        return (
            float(loss),
            len(self.testloader.dataset),
            {"loss": loss, "accuracy": accuracy, "dataset_name": self.dataset_name},
        )


class FedBNFlowerClient(FlowerClient):
    """Similar to FlowerClient but this is used by FedBN clients."""

    def __init__(self, save_path: Path, client_id: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # For FedBN clients we need to persist the state of the BN
        # layers across rounds. In Simulation clients are statess
        # so everything not communicated to the server (as it is the
        # case as with params in BN layers of FedBN clients) is lost
        # once a client completes its training. An upcoming version of
        # Flower suports stateful clients
        bn_state_dir = save_path / "bn_states"
        bn_state_dir.mkdir(exist_ok=True)
        self.bn_state_pkl = bn_state_dir / f"client_{client_id}.pkl"

    def _save_bn_statedict(self) -> None:
        """Save contents of state_dict related to BN layers."""
        bn_state = {
            name: val.cpu().numpy()
            for name, val in self.model.state_dict().items()
            if "bn" in name
        }

        with open(self.bn_state_pkl, "wb") as handle:
            pickle.dump(bn_state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_bn_statedict(self) -> Dict[str, torch.tensor]:
        """Load pickle with BN state_dict and return as dict."""
        with open(self.bn_state_pkl, "rb") as handle:
            data = pickle.load(handle)
        bn_stae_dict = {k: torch.tensor(v) for k, v in data.items()}
        return bn_stae_dict

    def get_parameters(self, config) -> NDArrays:
        """Return model parameters as a list of NumPy ndarrays w or w/o using BN.

        layers.
        """
        # First update bn_state_dir
        self._save_bn_statedict()
        # Excluding parameters of BN layers when using FedBN
        return [
            val.cpu().numpy()
            for name, val in self.model.state_dict().items()
            if "bn" not in name
        ]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from a list of NumPy ndarrays Exclude the bn layer if.

        available.
        """
        keys = [k for k in self.model.state_dict().keys() if "bn" not in k]
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=False)

        # Now also load from bn_state_dir
        if self.bn_state_pkl.exists():  # It won't exist in the first round
            bn_state_dict = self._load_bn_statedict()
            self.model.load_state_dict(bn_state_dict, strict=False)


def gen_client_fn(
    client_data: List[Tuple[DataLoader, DataLoader, str]],
    client_cfg: DictConfig,
    model_cfg: DictConfig,
    save_path: Path,
) -> Callable[[str], FlowerClient]:
    """Return a function that will be called to instantiate the cid-th client."""

    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""
        # Instantiate model
        net = instantiate(model_cfg)

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader, valloader, dataset_name = client_data[int(cid)]
        return instantiate(
            client_cfg,
            model=net,
            trainloader=trainloader,
            testloader=valloader,
            dataset_name=dataset_name,
            save_path=save_path,
            client_id=int(cid),
        )

    return client_fn
