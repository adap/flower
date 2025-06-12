"""fedprox: A Flower Baseline."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.typing import NDArrays, Scalar

from .dataset import load_data
from .model import get_weights, instantiate_model, set_weights, test, train
from .utils import context_to_easydict


# pylint: disable=too-many-arguments
class FlowerClient(NumPyClient):  # pylint: disable=too-many-instance-attributes
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        net: torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        num_epochs: int,
        learning_rate: float,
        straggler_schedule: np.ndarray,
        configs: dict,
    ):  # pylint: disable=too-many-arguments
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.straggler_schedule = straggler_schedule
        self.net.to(self.device)
        self.configs = configs

    def fit(self, parameters, config):
        """Implement distributed fit function for a given client."""
        set_weights(self.net, parameters)

        # At each round check if the client is a straggler,
        # if so, train less epochs (to simulate partial work)
        # if the client is told to be dropped (e.g. because not using
        # FedProx in the server), the fit method returns without doing
        # training.
        # This method always returns via the metrics (last argument being
        # returned) whether the client is a straggler or not. This info
        # is used by strategies other than FedProx to discard the update.
        if (
            self.straggler_schedule[int(config["current_round"]) - 1]
            and self.num_epochs > 1
        ):
            num_epochs = np.random.randint(1, self.num_epochs)
            if self.configs.run_config.fit.drop_client:
                # return without doing any training.
                # The flag in the metric will be used to tell the strategy
                # to discard the model upon aggregation
                return (
                    get_weights(self.net),
                    len(self.trainloader),
                    {"is_straggler": True},
                )

        else:
            num_epochs = self.num_epochs

        train(
            self.net,
            self.trainloader,
            self.device,
            epochs=num_epochs,
            learning_rate=self.learning_rate,
            proximal_mu=float(self.configs.run_config.algorithm.mu),
        )

        return get_weights(self.net), len(self.trainloader), {"is_straggler": False}

    def evaluate(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[float, int, dict]:
        """Implement distributed evaluation for a given client."""
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


# pylint: disable=E1101
def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""
    # Load model and data
    configs = context_to_easydict(context)
    net = instantiate_model(config=configs.run_config)
    partition_id = int(configs.node_config.partition_id)
    num_partitions = int(configs.node_config.num_partitions)
    local_epochs = configs.run_config.algorithm.local_epochs
    num_rounds = configs.run_config.algorithm.num_server_rounds
    stragglers = configs.run_config.algorithm.stragglers_fraction
    straggler_schedule = np.transpose(
        np.random.choice([0, 1], size=(num_rounds, 1), p=[1 - stragglers, stragglers])
    )[0]
    trainloader, valloader = load_data(
        dataset_config=configs.run_config.dataset,
        partition_id=partition_id,
        num_partitions=num_partitions,
    )
    learning_rate = configs.run_config.algorithm.learning_rate

    # Return Client instance
    return FlowerClient(
        net,
        trainloader,
        valloader,
        local_epochs,
        learning_rate,
        straggler_schedule,
        configs=configs,
    ).to_client()


# pylint: enable=E1101

# Flower ClientApp
app = ClientApp(client_fn)
