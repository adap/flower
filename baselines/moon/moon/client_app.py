"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""

import copy
import os
from collections import OrderedDict
from typing import Dict, Tuple

import torch
from flwr.common.typing import NDArrays, Scalar, UserConfig
from flwr.common import Context
from flwr.client import Client, NumPyClient, ClientApp
from torch.utils.data import DataLoader

from moon.models import init_net, train_fedprox, train_moon
from moon.dataset_preparation import get_dataset, get_data_transforms, get_transforms_apply_fn


# pylint: disable=too-many-instance-attributes
class FlowerClient(NumPyClient):
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        net_id: int,
        run_config: UserConfig,
        trainloader: DataLoader,
        device: torch.device,
    ):  # pylint: disable=too-many-arguments
        self.net_id = net_id
        self.dataset = run_config['dataset-name']
        self.model = run_config['model-name']
        self.output_dim = run_config['model-output-dim']
        self.trainloader = trainloader
        self.device = device
        self.num_epochs = run_config['num-epochs']
        self.learning_rate = run_config['learning-rate']
        self.mu = run_config['mu']  # pylint: disable=invalid-name
        self.temperature = run_config['temperature']
        self.model_dir = run_config['model-dir']
        self.alg = run_config['alg']

        self.net = init_net(self.dataset, self.model, self.output_dim)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Change the parameters of the model using the given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        self.set_parameters(parameters)
        prev_net = init_net(self.dataset, self.model, self.output_dim)
        if not os.path.exists(os.path.join(self.model_dir, str(self.net_id))):
            prev_net = copy.deepcopy(self.net)
        else:
            # load previous model from model_dir
            prev_net.load_state_dict(
                torch.load(
                    os.path.join(self.model_dir, str(self.net_id), "prev_net.pt")
                )
            )
        global_net = init_net(self.dataset, self.model, self.output_dim)
        global_net.load_state_dict(self.net.state_dict())
        if self.alg == "moon":
            train_moon(
                self.net,
                global_net,
                prev_net,
                self.trainloader,
                self.num_epochs,
                self.learning_rate,
                self.mu,
                self.temperature,
                self.device,
            )
        elif self.alg == "fedprox":
            train_fedprox(
                self.net,
                global_net,
                self.trainloader,
                self.num_epochs,
                self.learning_rate,
                self.mu,
                self.device,
            )
        if not os.path.exists(os.path.join(self.model_dir, str(self.net_id))):
            os.makedirs(os.path.join(self.model_dir, str(self.net_id)))
        torch.save(
            self.net.state_dict(),
            os.path.join(self.model_dir, str(self.net_id), "prev_net.pt"),
        )
        return self.get_parameters({}), len(self.trainloader), {"is_straggler": False}


def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset_name = context.run_config['dataset-name']
    fds = get_dataset(dataset_name=dataset_name,
                      dirichlet_alpha=context.run_config['dirichlet-alpha'],
                      num_partitions=context.node_config['num-partitions'],
                      partition_by=context.run_config['dataset-partition-by'],)
    
    partition_id = context.node_config["partition-id"]
    train_partition = fds.load_partition(partition_id=partition_id)

    train_transforms, _ = get_data_transforms(dataset_name=dataset_name)
    transforms_fn = get_transforms_apply_fn(train_transforms)
    trainloader = DataLoader(train_partition.with_transform(transforms_fn),
                            batch_size=context.run_config['batch-size'], shuffle=True)
    
    return FlowerClient(
        partition_id,
        context.run_config,
        trainloader,
        device,
    ).to_client()


app = ClientApp(client_fn=client_fn)
