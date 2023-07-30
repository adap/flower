from logging import INFO
import argparse
import flwr as fl
from time import time
from flwr.common.logger import log
from flwr.common.typing import Scalar
from flwr.client import NumPyClient, InMemoryClientState
import ray
import torch
import torchvision
import numpy as np
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Callable, Optional, Tuple, List
from sim_utils.dataset_utils import get_cifar_10, do_fl_partitioning, get_dataloader
from sim_utils.utils import Net, train, test

from flwr.simulation.virtual_client_state_manager import InFileSystemVirtualClientStateManager, SimpleVirtualClientStateManager


parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")

parser.add_argument("--num_client_cpus", type=int, default=1)
parser.add_argument("--num_rounds", type=int, default=5)
parser.add_argument("--state_in_fs", action="store_true", help='Records/loads client states to/from the file system')


# Flower client, adapted from Pytorch quickstart example
class FlowerClient(NumPyClient, InMemoryClientState):
    """A standard client that wants to record its state across rounds."""

    def __init__(self, cid: str, fed_dir_data: str):
        super().__init__()
        self.cid = cid
        self.fed_dir = Path(fed_dir_data)
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}

        # Instantiate model
        self.net = Net()

        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_parameters(self, config):
        return get_params(self.net)

    def fit(self, parameters, config):
        # This fit() method showcases how you can make use of the client's state. In this
        # simple example, we use the state to keep track of the time it takes to perform
        # the majority of the fit() operations. 
        # First, we display the state at the beginning of fit (state will be empty the first
        # time this client participates unless you initialise the state from disk)
        # Then, before ending fit() we show the updated state.

        set_params(self.net, parameters)

        # Load data for this client and get trainloader
        log(INFO, f"fit begins --> {self.fetch_state() = } for client: {self.cid}")
        t_start = time()
        num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
        trainloader = get_dataloader(
            self.fed_dir,
            self.cid,
            is_train=True,
            batch_size=config["batch_size"],
            workers=num_workers,
        )

        # Send model to device
        self.net.to(self.device)

        # Train
        train(self.net, trainloader, epochs=config["epochs"], device=self.device)
        self.update_state({'time_fit':time()-t_start})
        log(INFO, f"fit ends --> {self.fetch_state() = } for client: {self.cid}")

        # Return local model and statistics
        return get_params(self.net), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
        valloader = get_dataloader(
            self.fed_dir, self.cid, is_train=False, batch_size=50, workers=num_workers
        )

        # Send model to device
        self.net.to(self.device)

        # Evaluate
        loss, accuracy = test(self.net, valloader, device=self.device)

        # Return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}


def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 1,  # number of local epochs
        "batch_size": 64,
    }
    return config


def get_params(model: torch.nn.ModuleList) -> List[np.ndarray]:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_params(model: torch.nn.ModuleList, params: List[np.ndarray]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_evaluate_fn(
    testset: torchvision.datasets.CIFAR10,
) -> Callable[[fl.common.NDArrays], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
        server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""

        # determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = Net()
        set_params(model, parameters)
        model.to(device)

        testloader = torch.utils.data.DataLoader(testset, batch_size=50)
        loss, accuracy = test(model, testloader, device=device)

        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate


if __name__ == "__main__":
    # parse input arguments
    args = parser.parse_args()

    pool_size = 10  # number of dataset partitions (= number of total clients)
    client_resources = {
        "num_cpus": args.num_client_cpus
    }  # each client will get allocated 1 CPUs

    # Download CIFAR-10 dataset
    train_path, testset = get_cifar_10()

    # partition dataset (use a large `alpha` to make it IID;
    # a small value (e.g. 1) will make it non-IID)
    # This will create a new directory called "federated": in the directory where
    # CIFAR-10 lives. Inside it, there will be N=pool_size sub-directories each with
    # its own train/set split.
    fed_dir = do_fl_partitioning(
        train_path, pool_size=pool_size, alpha=1000, num_classes=10, val_ratio=0.1
    )

    # configure the strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5,
        fraction_evaluate=0.0,
        min_fit_clients=5,
        min_evaluate_clients=0,
        min_available_clients=pool_size,  # All clients should be available
        on_fit_config_fn=fit_config,
        evaluate_fn=get_evaluate_fn(testset),  # centralised evaluation of global model
    )

    def client_fn(cid: str):
        return FlowerClient(cid, fed_dir_data=fed_dir)

    # (optional) specify Ray config
    ray_init_args = {"include_dashboard": False}

    # let's ensure we save the log to a file (note this will append to the same file each time)
    fl.common.logger.configure(identifier="mySimulation", filename="sim_stateful_log.txt")

    if args.state_in_fs:
        # Let's record the clients' state to the file system.
        # In simulation (because clients can run in different machines)
        # accessing the file system (FS) is done through the VirtualClientStateManager

        # Let's use a simple InFileSystemVirtualClientStateManager which stores the state
        # of each client individually as a standard python pickle
        state_fs = InFileSystemVirtualClientStateManager(state_dir="simulation_state_FS")
        #! If you run this example a second time, your clients will read their initial state
        # from their corresponding pickle file
    else:
        # else, use the default (in-memory state). This will ensure that your clients' state
        # is preserver throughout the entire simulation
        state_fs = SimpleVirtualClientStateManager()

    log(INFO, "-----------------------------------------------------------")
    log(INFO, f"Using VirtualClientStateManager of type: {type(state_fs)}")

    # start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=pool_size,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        ray_init_args=ray_init_args,
        state_manager=state_fs,
    )
