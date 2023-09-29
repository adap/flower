"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""

from collections import OrderedDict
from typing import Callable, Dict, List, Tuple, overload

import flwr as fl
import numpy as np
import torch
from flwr.common.typing import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from FedSMOO.dataset import load_datasets
from FedSMOO.models import *

# -------------------- FedSMOO Client ----------------------------------
class FlowerClientFedSMOO(
    fl.client.NumPyClient
):  # pylint: disable=too-many-instance-attributes
    """Standard Flower client for CNN training."""

    def __init__(self,
        net: torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        local_epochs: int,
        learning_rate: float,
        weight_decay: float,
        sch_step: int,
        sch_gamma: float,
        sam_lr: float,
        alpha: float, 
        lr_decay: float):  # pylint: disable=too-many-arguments
        
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.sch_step = sch_step
        self.sch_gamma = sch_gamma

        params = self.get_parameters({})
        self.hist_sam_diffs_list = [torch.zeros(mat.shape) for mat in params]   # mu - list[torch.tensor] (weight matrices)
        self.hist_params_diff =  get_H_param_array(self.hist_sam_diffs_list)    # lambda - numpy array (1d)
        self.sam_lr = sam_lr
        self.lr_decay = lr_decay 

        self.alpha_coef = alpha          
        # TODO: later set this to weighted by trainloader lengths
        # can be done by sending weights in config from the server

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Returns the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Changes the parameters of the model using the given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implements distributed evaluation for a given client."""
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
    
    def fit(
        self, parameters, # : NDArrays, 
        config #: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implements distributed fit function for a given client."""
            
        self.learning_rate = self.learning_rate * self.lr_decay
        self.set_parameters(parameters) # update params to global params

        gs_diff_list = config["gs_diff_list"] # server sent `s` NDArrays
        gs_diff_list_torch = [torch.from_numpy(t) for t in gs_diff_list]

        init_params_list = config["init_mdl_param"] # server sent NDArrays
        init_params_list_torch = [torch.from_numpy(t) for t in init_params_list]
        init_params_torch1d = torch.tensor(get_H_param_array(init_params_list_torch))

        # returns a torch net, list of torch tensors
        net, self.hist_sam_diffs_list = trainFedSMOO(self.net,
                                    self.trainloader,
                                    self.device,
                                    self.alpha_coef,
                                    init_params_torch1d, # server sent NDArrays -> torch 1D tensor
                                    torch.tensor(self.hist_params_diff),     # lambda
                                    self.hist_sam_diffs_list,  # mu
                                    gs_diff_list_torch, # server sent
                                    self.local_epochs,              
                                    self.learning_rate,
                                    self.weight_decay,
                                    self.sch_step,
                                    self.sch_gamma,
                                    self.sam_lr,)    # r in the algo
        
        curr_model_params = get_mdl_params([net])[0] # 1d numpy array
        prev_model_params = get_mdl_params([self.net]) [0]
        self.hist_params_diff += curr_model_params - prev_model_params
        self.net = net

        # if returning more than just the model weights 
        # in 1st return value, 
        # unpack the same way in fit_aggregate
        return self.get_parameters({}), len(self.trainloader), {"hist_sam_diffs": [get_H_param_array(self.hist_sam_diffs_list)], 
                                                                "is_straggler": False}


# -------------------- FedAvg Client -----------------------------------
class FlowerClientFedAvg(fl.client.NumPyClient):
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        net: torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        local_epochs: int,
        learning_rate: float,
        lr_decay: float,
        straggler_schedule: np.ndarray,
    ):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.straggler_schedule = straggler_schedule

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Returns the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Changes the parameters of the model using the given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implements distributed fit function for a given client."""
        
        self.set_parameters(parameters)
        self.learning_rate = self.learning_rate*self.lr_decay

        # if (
            # self.straggler_schedule[int(config["curr_round"]) - 1]
        # ):
            # # return without doing any training.
            # # The flag in the metric will be used to tell the strategy
            # # to discard the model upon aggregation
            # return (
                # [],
                # len(self.trainloader),
                # {"is_straggler": True},
            # )
        
        trainFedAvg(
            self.net,
            self.trainloader,
            self.device,
            epochs=self.local_epochs,
            learning_rate=self.learning_rate,
        )
        return self.get_parameters({}), len(self.trainloader), {"is_straggler": False}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implements distributed evaluation for a given client."""
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


# -------------------- FedDyn Client -----------------------------------


# ---------------- Generate Client Function FedSMOO ------------------------

def gen_client_fn_FedSMOO(
    local_epochs: int,
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    learning_rate: float,
    model: DictConfig,
    weight_decay: float,
    sch_step: int,
    sch_gamma: float,
    sam_lr: float,
    alpha: float,
    lr_decay: float,
) -> Callable[[str], FlowerClientFedSMOO]:  # pylint: disable=too-many-arguments
    
    """Generates the client function that creates the Flower Clients.

    Parameters
    ----------
    local_epochs : int
        The number of local epochs each client should run the training for before
        sending it to the server.
    trainloaders: List[DataLoader]
        A list of DataLoaders, each pointing to the dataset training partition
        belonging to a particular client.
    valloaders: List[DataLoader]
        A list of DataLoaders, each pointing to the dataset validation partition
        belonging to a particular client.
    learning_rate : float
        The learning rate for the SGD optimizer of clients.
    model: DictConfig
        config dict to instantiate a model
    weight_decay: float
        SGD weight decay
    sch_step: int
        Number of sceduler steps before LR update
    sch_gamma: float
        scheduler gamma factor for lr multiplication
    sam_lr: float
        sam lr
    alpha: float
        client momentum term alpha
    lr_decay: float
        round wise learning rate decay

    Returns
    -------
    Callable[[str], FlowerClient]
        the client function that creates Flower Clients
    """

    def client_fn(cid: str) -> FlowerClientFedSMOO:
        """Create a Flower client representing a single organization."""

        # Load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = instantiate(model).to(device)

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]

        return FlowerClientFedSMOO(
            net,
            trainloader,
            valloader,
            device,
            local_epochs,
            learning_rate,
            weight_decay,
            sch_step,
            sch_gamma,
            sam_lr, 
            alpha, 
            lr_decay)

    return client_fn

# ---------------- Generate Client Function FedAvg -------------------------

def gen_client_fn_FedAvg(
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    local_epochs: int,
    learning_rate: float,
    num_rounds: int,
    num_clients: int,
    stragglers: float,
    lr_decay: float,
    model: DictConfig,
) -> Callable[[str], FlowerClientFedAvg]:
    """Generates the client function that creates the Flower Clients.

    Parameters
    ----------
    num_clients : int
        The number of clients present in the setup
    local_epochs : int
        The number of local epochs each client should run the training for before
        sending it to the server.
    batch_size : int
        The size of the local batches each client trains on.
    learning_rate : float
        The learning rate for the SGD  optimizer of clients.

    Returns
    -------
    Callable[[str], FlowerClientFedAvg]
        A tuple containing the client function that creates Flower Clients and
        the DataLoader that will be used for testing
    """

    stragglers_mat = np.transpose(
        np.random.choice(
            [0, 1], size=(num_rounds, num_clients), p=[1 - stragglers, stragglers]
        )
    )

    def client_fn(cid: str) -> FlowerClientFedAvg:
        """Create a Flower client representing a single organization."""

        # Load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = instantiate(model).to(device)

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]

        # Create a  single Flower client representing a single organization
        return FlowerClientFedAvg(
            net, trainloader, valloader, device, 
            local_epochs, learning_rate, lr_decay, stragglers_mat[int(cid)],
        )

    return client_fn

# ---------------- Generate Client Function FedDyn ------------------------
