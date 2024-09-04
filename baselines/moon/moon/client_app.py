"""moon: A Flower Baseline."""

from collections import OrderedDict
from typing import Dict, Tuple

import torch
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader

from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Array, Context, ParametersRecord, array_from_numpy
from flwr.common.typing import NDArrays, Scalar
from moon.dataset_preparation import get_data_transforms, get_transforms_apply_fn
from moon.models import init_net, train_fedprox, train_moon


# pylint: disable=too-many-instance-attributes
class FlowerClient(NumPyClient):
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        context: Context,
        trainloader: DataLoader,
        device: torch.device,
    ):  # pylint: disable=too-many-arguments
        # Pin the state received from the Context object
        # It will be used to persist the model parameters of this client
        self.client_state = context.state
        self.local_model_name = "prev_net"
        run_config = context.run_config
        self.dataset = run_config["dataset-name"]
        self.model = run_config["model-name"]
        self.output_dim = run_config["model-output-dim"]
        self.trainloader = trainloader
        self.device = device
        self.num_epochs = run_config["num-epochs"]
        self.learning_rate = run_config["learning-rate"]
        self.mu = run_config["mu"]  # pylint: disable=invalid-name
        self.temperature = run_config["temperature"]
        self.alg = run_config["alg"]

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
        if self.alg == "moon":
            prev_net = init_net(self.dataset, self.model, self.output_dim)
            # If `prev_net` key found in this client's state (meaning this is not the
            # first time this client participates), use the previously saved parameters
            if self.local_model_name in self.client_state.parameters_records:
                self.load_model_from_context(prev_net)

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
        if self.alg == "moon":
            # Save current model parameters so they can be used next time
            # this client is sampled to participate in a round
            self.save_local_model_to_context()
        return self.get_parameters({}), len(self.trainloader), {"is_straggler": False}

    def save_local_model_to_context(self) -> None:
        """Save the state_dict of model to this client's context."""
        state_dict_arrays: OrderedDict[str, Array] = OrderedDict()
        for k, v in self.net.state_dict().items():
            state_dict_arrays[k] = array_from_numpy(v.cpu().numpy())

        # Add to recordset (replace if already exists)
        self.client_state.parameters_records[self.local_model_name] = ParametersRecord(
            state_dict_arrays
        )

    def load_model_from_context(self, model: torch.nn.Module) -> None:
        """Reconstruct PyTorch state_dict from context."""
        state_dict = {}
        for k, v in self.client_state.parameters_records[self.local_model_name].items():
            state_dict[k] = torch.from_numpy(v.numpy())

        # Apply loaded state_dict to model
        model.load_state_dict(state_dict, strict=True)


FDS = None


def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset_name = context.run_config["dataset-name"]
    partition_by = context.run_config["dataset-partition-by"]

    # Only initialize `FederatedDataset` once
    global FDS  # pylint: disable=global-statement
    if FDS is None:
        partitioner = DirichletPartitioner(
            num_partitions=context.node_config["num-partitions"],
            alpha=context.run_config["dirichlet-alpha"],
            partition_by=partition_by,
            seed=int(context.run_config["seed"]),
        )
        FDS = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train": partitioner},
        )

    partition_id = int(context.node_config["partition-id"])
    train_partition = FDS.load_partition(partition_id=partition_id)
    train_partition.set_format("torch")

    train_transforms, _ = get_data_transforms(dataset_name=dataset_name)
    transforms_fn = get_transforms_apply_fn(train_transforms, partition_by)
    trainloader = DataLoader(
        train_partition.with_transform(transforms_fn),
        batch_size=context.run_config["batch-size"],
        drop_last=True,
        shuffle=True,
    )

    return FlowerClient(
        context,
        trainloader,
        device,
    ).to_client()


app = ClientApp(client_fn=client_fn)
