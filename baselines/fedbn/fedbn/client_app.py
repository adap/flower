"""fedbn: A Flower Baseline."""

from typing import OrderedDict

import numpy as np
import torch

from fedbn.dataset import get_data
from fedbn.model import CNNModel, test, train
from fedbn.utils import extract_weights
from flwr.client import ClientApp, NumPyClient
from flwr.common import Array, ArrayRecord, Context, NDArrays


class FlowerClient(NumPyClient):
    """A class defining the client."""

    # pylint: disable=unused-argument
    def __init__(
        self,
        net,
        trainloader,
        testloader,
        dataset_name,
        learning_rate,
        **kwargs,
    ):
        self.trainloader = trainloader
        self.testloader = testloader
        self.dataset_name = dataset_name
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.net = net.to(self.device)
        self.learning_rate = learning_rate

    def get_weights(self) -> NDArrays:
        """Return model parameters as a list of NumPy ndarrays w or w/o.

        using BNlayers.
        """
        # Return all model parameters as a list of NumPy ndarrays
        return extract_weights(self.net, "FedAvg")

    def set_weights(self, parameters: NDArrays) -> None:
        """Set model parameters from a list of NumPy ndarrays Exclude the BN.

        layer if available.
        """
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train model using this client's data."""
        self.set_weights(parameters)
        pre_train_loss, pre_train_acc = test(
            self.net, self.trainloader, device=self.device
        )
        # Train model on local dataset
        loss, acc = train(
            self.net,
            self.trainloader,
            epochs=1,
            learning_rate=self.learning_rate,
            device=self.device,
        )
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
            self.get_weights(),
            len(self.trainloader),
            metrics,
        )

    def evaluate(self, parameters, config):
        """Evaluate model using this client's data."""
        self.set_weights(parameters)
        loss, accuracy = test(self.net, self.testloader, device=self.device)
        return (
            float(loss),
            len(self.testloader),
            {
                "loss": loss,
                "accuracy": accuracy,
                "dataset_name": self.dataset_name,
            },
        )


class FedBNFlowerClient(FlowerClient):
    """Similar to FlowerClient but this is used by FedBN clients."""

    def __init__(self, client_state, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.client_state = client_state
        # For FedBN clients we need to persist the state of the BN
        # layers across rounds. In Simulation clients are states
        # so everything not communicated to the server (as it is the
        # case as with params in BN layers of FedBN clients) is lost
        # once a client completes its training. This is the case unless
        # we preserve the batch norm states in the Context.
        if not self.client_state.array_records:
            # Ensure statefulness of error feedback buffer.
            self.client_state.array_records["local_batch_norm"] = ArrayRecord(
                OrderedDict({"initialisation": Array(np.array([-1]))})
            )

    def _save_bn_statedict(self) -> None:
        """Save contents of state_dict related to BN layers."""
        bn_state = OrderedDict(
            {
                name: Array(val.cpu().numpy())
                for name, val in self.net.state_dict().items()
                if "bn" in name
            }
        )
        self.client_state.array_records["local_batch_norm"] = ArrayRecord(
            bn_state
        )

    def get_weights(self) -> NDArrays:
        """Return model parameters as a list of NumPy ndarrays without BN.

        layers.
        """
        # First update bn_state_dir
        self._save_bn_statedict()
        return extract_weights(self.net, "FedBN")

    def set_weights(self, parameters: NDArrays) -> None:
        """Set model parameters from a list of NumPy ndarrays Exclude the bn.

        layer if available.
        """
        keys = [k for k in self.net.state_dict().keys() if "bn" not in k]
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

        # Now also load from bn_state_dir
        if (
            "initialisation"
            not in self.client_state.array_records["local_batch_norm"].keys()
        ):  # It won't exist in the first round
            batch_norm_state = {
                k: torch.tensor(v.numpy())
                for k, v in self.client_state.array_records[
                    "local_batch_norm"
                ].items()
            }
            self.net.load_state_dict(batch_norm_state, strict=True)


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""
    # Load model and data
    run_config = context.run_config
    net = CNNModel(num_classes=run_config["num-classes"])
    partition_id = int(context.node_config["partition-id"])
    trainloader, valloader, dataset_name = (get_data(context))[partition_id]

    # Return Client instance
    client_type, client_state = (
        (FlowerClient, None)
        if run_config["algorithm-name"] == "FedAvg"
        else (FedBNFlowerClient, context.state)
    )
    return client_type(
        net=net,
        trainloader=trainloader,
        testloader=valloader,
        dataset_name=dataset_name,
        learning_rate=run_config["learning-rate"],
        client_state=client_state,
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
